#!/usr/bin/env python3
"""Train a room-presence head on frozen contour features."""

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
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.room_mask_decoder import room_mask_targets
from floorflow.heads.room_program import RoomProgramHead, room_presence_target, room_program_loss
from floorflow.heads.room_program_set import (
    RoomProgramSetHead,
    load_room_program_set_state,
    program_set_metrics,
    room_program_set_loss,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-head", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/room_program")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--head-type", choices=("bce", "set"), default="bce")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--no-object-weight", type=float, default=0.1)
    parser.add_argument("--cardinality-weight", type=float, default=0.5)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--cooccurrence-weight", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        choices=("presence_f1", "exact_set_accuracy", "composite"),
        default="presence_f1",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--train-presence-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def features_for(backbone, footprint, floor_ids):
    zero = torch.zeros(
        footprint.shape[0],
        backbone.target_channels,
        *footprint.shape[-2:],
        device=footprint.device,
    )
    with torch.no_grad():
        return backbone._encode_features(
            zero,
            torch.zeros(footprint.shape[0], device=footprint.device),
            footprint,
            floor_ids=floor_ids,
        )


def presence_metrics(logits, labels, taxonomy) -> dict[str, float]:
    target = room_presence_target(labels, taxonomy)
    predicted = logits.sigmoid() > 0.5
    expected = target.bool()
    tp = (predicted & expected).sum().float()
    precision = tp / predicted.sum().clamp_min(1.0)
    recall = tp / expected.sum().clamp_min(1.0)
    return {
        "presence_precision": float(precision.cpu()),
        "presence_recall": float(recall.cpu()),
        "presence_f1": float((2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6)).cpu()),
        "exact_set_accuracy": float((predicted == expected).all(dim=1).float().mean().cpu()),
    }


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
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    head_type = RoomProgramSetHead if args.head_type == "set" else RoomProgramHead
    head = head_type(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        hidden_dim=args.hidden_dim,
        tokens=args.tokens,
        layers=args.layers,
        heads=args.heads,
        **({"num_queries": args.queries} if args.head_type == "set" else {}),
    ).to(device)
    if args.init_head is not None:
        if args.head_type != "set":
            raise SystemExit("--init-head requires --head-type set")
        initial = torch.load(
            args.init_head,
            map_location=device,
            weights_only=False,
        )
        load_room_program_set_state(head, initial["head"])
        head.class_presence_enabled.fill_(True)
    if args.train_presence_only:
        if args.head_type != "set":
            raise SystemExit("--train-presence-only requires --head-type set")
        for name, parameter in head.named_parameters():
            parameter.requires_grad_(
                name.startswith(
                    (
                        "presence_",
                        "query_cardinality_residual.",
                        "cardinality_head.",
                    )
                )
                or name == "class_presence_enabled"
            )
    positive = torch.from_numpy(
        load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
    ).to(device)
    room_weights = positive[torch.tensor([class_id - 1 for class_id in taxonomy.room_ids], device=device)]
    room_weights = room_weights / room_weights.mean()
    if args.head_type == "set":
        head.program_prior.copy_(room_weights.detach())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1.0e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        head.train()
        total = 0.0
        seen = 0
        component_totals: dict[str, float] = {}
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            features = features_for(backbone, footprint, floor_ids.to(device))
            predictions = head(features)
            if args.head_type == "set":
                target_classes, _target_masks, target_valid = room_mask_targets(
                    target,
                    taxonomy,
                    max_rooms=args.queries,
                )
                loss, components = room_program_set_loss(
                    predictions,
                    target_classes,
                    target_valid,
                    class_weights=room_weights,
                    no_object_weight=args.no_object_weight,
                    cardinality_weight=args.cardinality_weight,
                    presence_weight=args.presence_weight,
                    cooccurrence_weight=args.cooccurrence_weight,
                )
            else:
                loss, components = room_program_loss(
                    predictions,
                    target,
                    taxonomy,
                    class_weights=room_weights,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * target.shape[0]
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + float(
                    value.detach().cpu()
                ) * target.shape[0]
            seen += target.shape[0]
        head.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target = target.to(device)
                footprint = footprint.to(device)
                outputs.append(head(features_for(backbone, footprint, floor_ids.to(device))))
                labels.append(target)
        target_presence = torch.cat(
            [room_presence_target(label, taxonomy) for label in labels]
        ).bool()
        if args.head_type == "set":
            predictions = {
                key: torch.cat([output[key] for output in outputs])
                for key in (
                    "class_logits",
                    "cardinality_logits",
                    "presence_logits",
                    "presence_rank_logits",
                )
            }
            metrics = program_set_metrics(predictions, target_presence)
        else:
            metrics = presence_metrics(
                torch.cat([output["presence_logits"] for output in outputs]),
                torch.cat(labels),
                taxonomy,
            )
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            **{
                f"train_{key}": value / max(1, seen)
                for key, value in component_totals.items()
            },
            "head_type": args.head_type,
            **metrics,
        }
        if args.selection_metric == "composite":
            if args.head_type == "set":
                selection_score = (
                    record["presence_f1"]
                    + record["exact_set_accuracy"]
                    + record["cardinality_accuracy"]
                )
            else:
                selection_score = record["presence_f1"] + record["exact_set_accuracy"]
        else:
            selection_score = record[args.selection_metric]
        record["selection_metric"] = args.selection_metric
        record["selection_score"] = selection_score
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "head": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "backbone_checkpoint": str(args.checkpoint),
            "head_type": args.head_type,
            "config": {**vars(args), "image_size": int(config.get("image_size", 256))},
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if args.save_every_epoch:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch:03d}.pt")
        if selection_score > best:
            best = selection_score
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
