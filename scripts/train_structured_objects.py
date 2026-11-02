#!/usr/bin/env python3
"""Train independent wall, opening, and furniture heads on frozen features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.class_balance import load_positive_weights
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.structured_objects import (
    StructuredObjectHeads,
    group_ce_dice_loss,
    initialize_object_priors,
    object_group_metrics,
    object_group_predictions,
    wall_bce_dice_loss,
)
from floorflow.heads.geometry_heads import opening_candidate_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/structured_objects")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--opening-weight", type=float, default=1.0)
    parser.add_argument("--opening-candidate-weight", type=float, default=0.5)
    parser.add_argument("--furniture-weight", type=float, default=1.0)
    parser.add_argument("--wall-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def features_for(backbone, target, footprint, floor_ids):
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
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, persistent_workers=args.workers > 0, **loader_args)
    val_loader = DataLoader(val, shuffle=False, persistent_workers=args.workers > 0, **loader_args)
    head = StructuredObjectHeads(backbone.output.in_channels, taxonomy).to(device)
    if args.initialize_from is not None:
        payload = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        initial = payload.get("head", payload.get("model", payload))
        current = head.state_dict()
        compatible = {
            key.removeprefix("structured_objects."): value
            for key, value in initial.items()
            if key.removeprefix("structured_objects.") in current
            and current[key.removeprefix("structured_objects.")].shape == value.shape
        }
        head.load_state_dict(compatible, strict=False)
        print(f"initialized {len(compatible)}/{len(current)} object tensors", flush=True)
    else:
        frequency_path = args.cache_root / "class_frequency.json"
        if frequency_path.is_file():
            frequency = json.loads(frequency_path.read_text(encoding="utf-8"))
            print(
                f"initialized object priors: {initialize_object_priors(head, frequency['pixel_counts'])}",
                flush=True,
            )
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    positive_weights = torch.from_numpy(
        load_positive_weights(args.cache_root, taxonomy.num_target_channels, 20.0)
    ).to(device)
    opening_weights = positive_weights[[index - 1 for index in taxonomy.opening_ids]]
    furniture_weights = positive_weights[[index - 1 for index in taxonomy.furniture_ids]]
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device)
    wall_id = taxonomy.class_id("wall")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        head.train()
        totals = {
            "loss": 0.0,
            "opening_loss": 0.0,
            "opening_candidate_loss": 0.0,
            "furniture_loss": 0.0,
            "wall_loss": 0.0,
        }
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
            features = features_for(backbone, target, footprint, floor_ids)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                outputs = head(features)
            outputs = {key: value.float() for key, value in outputs.items()}
            opening_loss, _, _ = group_ce_dice_loss(
                outputs["opening_logits"],
                target,
                opening_ids,
                footprint,
                opening_weights,
                outputs["opening_background_logits"],
            )
            furniture_loss, _, _ = group_ce_dice_loss(
                outputs["furniture_logits"],
                target,
                furniture_ids,
                footprint,
                furniture_weights,
                outputs["furniture_background_logits"],
            )
            wall_loss = wall_bce_dice_loss(outputs["wall_logits"], target, wall_id, footprint)
            opening_candidate, _ = opening_candidate_loss(
                outputs,
                target,
                footprint,
                taxonomy,
            )
            loss = (
                args.opening_weight * opening_loss
                + args.opening_candidate_weight * opening_candidate
                + args.furniture_weight * furniture_loss
                + args.wall_weight * wall_loss
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            values = {
                "loss": loss,
                "opening_loss": opening_loss,
                "opening_candidate_loss": opening_candidate,
                "furniture_loss": furniture_loss,
                "wall_loss": wall_loss,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        head.eval()
        rows = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
                outputs = head(features_for(backbone, target, footprint, floor_ids))
                predictions = object_group_predictions(
                    outputs,
                    taxonomy,
                    opening_threshold=0.5,
                    furniture_threshold=0.5,
                    score_scale="probability",
                )
                rows.append(object_group_metrics(predictions, target, taxonomy))
        metrics = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        record = {**{key: value / max(1, seen) for key, value in totals.items()}, **metrics, "epoch": epoch}
        print(json.dumps(record), flush=True)
        with (args.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "head": head.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "backbone_checkpoint": str(args.checkpoint),
            "config": {**vars(args), "image_size": int(config.get("image_size", 256))},
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        quality = metrics["furniture_f1"] + metrics["opening_f1"] + metrics["wall_iou"]
        if quality > best:
            best = quality
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
