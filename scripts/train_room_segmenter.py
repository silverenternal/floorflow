#!/usr/bin/env python3
"""Train and validate a direct contour-to-room segmentation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from floorflow.data.class_balance import load_positive_weights
from floorflow.data.data_balance import (
    cached_class_presence_weights,
    make_weighted_sampler,
)
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.room_groups import room_group_macro_iou
from floorflow.heads.room_partition import (
    batch_room_partition_target,
    room_partition_loss,
)
from floorflow.heads.room_segmenter import ContourRoomSegmenter
from floorflow.objectives.condition_objective import (
    partition_boundary_regularizer,
    room_macro_iou_from_labels,
)
from floorflow.training.crop_curriculum import random_crop_and_resize
from floorflow.training.training import ModelEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/room_segmenter")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--sampler",
        choices=("uniform", "class-balanced"),
        default="uniform",
        help="Preserve uniform visitation or oversample room-rare cases.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--preload-data", action="store_true")
    parser.add_argument("--initialize-checkpoint", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--crop-consistency-probability", type=float, default=0.0)
    parser.add_argument("--crop-minimum-fraction", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, taxonomy, device) -> dict[str, float]:
    model.eval()
    rows = []
    for target, footprint, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        logits = model(footprint)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        prediction = room_ids[logits.argmax(dim=1)] * footprint.long()
        for index in range(target.shape[0]):
            rows.append(
                {
                    "room_macro_iou": room_macro_iou_from_labels(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        taxonomy,
                    ),
                    "room_group_macro_iou": room_group_macro_iou(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        taxonomy,
                    ),
                }
            )
    return {
        key: sum(float(row[key]) for row in rows) / max(1, len(rows))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
        preload=args.preload_data,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        preload=args.preload_data,
    )
    taxonomy = train.taxonomy
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 4 if args.workers > 0 else None,
    }
    if args.sampler == "class-balanced":
        sample_weights = cached_class_presence_weights(
            train,
            taxonomy.room_ids,
            power=0.5,
        )
        train_loader = DataLoader(
            train,
            sampler=make_weighted_sampler(sample_weights),
            **common,
        )
    else:
        train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    model = ContourRoomSegmenter(
        len(taxonomy.room_ids),
        pretrained=not args.no_pretrained,
        decoder_channels=args.decoder_channels,
    ).to(device)
    if args.initialize_checkpoint is not None:
        payload = torch.load(
            args.initialize_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state = payload.get("raw_model", payload.get("model"))
        if state is None:
            raise ValueError("initialization checkpoint has no model weights")
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) * args.epochs),
        eta_min=args.lr * 0.1,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=args.ema_decay)
    positive = torch.from_numpy(
        load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
    ).to(device)
    room_weights = positive[torch.tensor([class_id - 1 for class_id in taxonomy.room_ids], device=device)]
    room_weights = room_weights / room_weights.mean()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "partition": 0.0, "boundary": 0.0}
        seen = 0
        for target, footprint, _floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            if args.crop_consistency_probability > 0:
                target, footprint = random_crop_and_resize(
                    target,
                    footprint,
                    probability=args.crop_consistency_probability,
                    minimum_fraction=args.crop_minimum_fraction,
                )
            partition_target = batch_room_partition_target(
                target, footprint, taxonomy, fill_occluded=False
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(footprint)
                partition = room_partition_loss(
                    logits,
                    partition_target,
                    class_weights=room_weights,
                )["loss"]
                boundary = partition_boundary_regularizer(logits, partition_target)
                loss = partition + args.boundary_weight * boundary
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            totals["loss"] += float(loss.detach().cpu()) * target.shape[0]
            totals["partition"] += float(partition.detach().cpu()) * target.shape[0]
            totals["boundary"] += float(boundary.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        metrics = evaluate(ema.module, val_loader, taxonomy, device)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(1, seen) for key, value in totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model": ema.state_dict(),
            "raw_model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": {
                **vars(args),
                "model": "resnet18_fpn_contour_room_segmenter",
                "objective": "dense_room_partition",
            },
            "epoch": epoch,
            "val": record,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if metrics["room_macro_iou"] > best:
            best = metrics["room_macro_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
