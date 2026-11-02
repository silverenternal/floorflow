#!/usr/bin/env python3
"""Train a dense room-partition head on a frozen contour backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.inference.retrieval_prior import ContourRoomRetriever, prior_one_hot
from floorflow.heads.room_partition import (
    RoomPartitionHead,
    batch_room_partition_target,
    room_partition_loss,
    room_partition_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/room_partition")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--retrieval-top-k", type=int, default=0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def class_weights(taxonomy, power: float, maximum: float, device: torch.device) -> torch.Tensor:
    source_counts = taxonomy.source_counts.get("room", {})
    counts = []
    for class_id in taxonomy.room_ids:
        class_name = taxonomy.class_names[class_id]
        counts.append(
            max(
                1,
                sum(
                    int(count)
                    for subtype, target_name in taxonomy.room_map.items()
                    if target_name == class_name
                    for count in [source_counts.get(subtype, 0)]
                ),
            )
        )
    counts = torch.tensor(counts, dtype=torch.float32, device=device)
    weights = (counts / counts.sum()).pow(-power).clamp(0.5, maximum)
    return weights / weights.mean()


def forward_features(backbone, target, footprint, floor_ids):
    zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=target.device)
    with torch.no_grad():
        return backbone._encode_features(
            zero,
            torch.zeros(target.shape[0], device=target.device),
            footprint,
            floor_ids=floor_ids,
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    backbone, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    backbone.requires_grad_(False)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=True,
        limit=args.limit_train,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
    )
    loader = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, persistent_workers=args.workers > 0, **loader)
    val_loader = DataLoader(val, shuffle=False, persistent_workers=args.workers > 0, **loader)
    retriever = None
    train_names = None
    if args.retrieval_top_k > 0:
        train.augment = False
        train.preload()
        retriever = ContourRoomRetriever(train._targets, train._footprints, taxonomy, args.retrieval_top_k)
        train_names = {
            f"{sample.relative_case}__{sample.floor_id}": index
            for index, sample in enumerate(train.samples)
        }
    head = RoomPartitionHead(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        prior_channels=len(taxonomy.room_ids) if retriever is not None else 0,
    ).to(device)
    if args.initialize_from is not None:
        initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)["head"]
        current = head.state_dict()
        compatible = {key: value for key, value in initial.items() if key in current and current[key].shape == value.shape}
        head.load_state_dict(compatible, strict=False)
        print(f"initialized {len(compatible)}/{len(current)} partition tensors", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    weights = class_weights(taxonomy, args.class_weight_power, args.max_class_weight, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(args.epochs):
        head.train()
        totals = {"loss": 0.0, "partition_ce": 0.0, "partition_dice": 0.0}
        seen = 0
        for target, footprint, floor_ids, names in train_loader:
            target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
            features = forward_features(backbone, target, footprint, floor_ids)
            partition_target = batch_room_partition_target(
                target, footprint, taxonomy, fill_occluded=False
            )
            prior = None
            if retriever is not None:
                exclude = np.asarray([train_names[name] for name in names])
                prior = prior_one_hot(
                    retriever.retrieve(footprint.cpu().numpy(), exclude_indices=exclude),
                    len(taxonomy.room_ids),
                    device,
                )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = head(features, prior)
                losses = room_partition_loss(logits, partition_target, weights, args.dice_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        head.eval()
        metrics = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
                features = forward_features(backbone, target, footprint, floor_ids)
                partition_target = batch_room_partition_target(
                    target, footprint, taxonomy, fill_occluded=False
                )
                prior = None
                if retriever is not None:
                    prior = prior_one_hot(
                        retriever.retrieve(footprint.cpu().numpy()),
                        len(taxonomy.room_ids),
                        device,
                    )
                logits = head(features, prior)
                metrics.append(room_partition_metrics(logits, partition_target, taxonomy))
        record = {
            "epoch": epoch,
            **{key: value / max(1, seen) for key, value in totals.items()},
            **{key: sum(row[key] for row in metrics) / max(1, len(metrics)) for key in metrics[0]},
        }
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "head": head.state_dict(),
            "backbone_checkpoint": str(args.checkpoint),
            "taxonomy": taxonomy.to_json(),
            "config": {
                **vars(args),
                "image_size": int(config.get("image_size", 256)),
                "backbone_stem_channels": int(backbone.condition_encoder.stem.weight.shape[0]),
                "retrieval_top_k": args.retrieval_top_k,
            },
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if record["room_partition_iou"] > best_iou:
            best_iou = record["room_partition_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
