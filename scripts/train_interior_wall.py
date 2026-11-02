#!/usr/bin/env python3
"""Train an interior wall head on frozen contour backbone features."""

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
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.geometry.interior_wall import InteriorWallHead, interior_wall_iou, interior_wall_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/interior_wall")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--target-dilation", type=int, default=1)
    parser.add_argument("--dilation-anneal-fraction", type=float, default=0.5)
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
        limit=args.limit_train,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit_val,
    )
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train, shuffle=True, persistent_workers=args.workers > 0, **common)
    val_loader = DataLoader(val, shuffle=False, persistent_workers=args.workers > 0, **common)
    head = InteriorWallHead(backbone.output.in_channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        head.train()
        totals = {"loss": 0.0, "bce": 0.0, "tversky": 0.0}
        seen = 0
        dilation = (
            max(0, args.target_dilation)
            if epoch < int(args.epochs * args.dilation_anneal_fraction)
            else 0
        )
        for target, footprint, floor_ids, _names in train_loader:
            target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
            features = features_for(backbone, target, footprint, floor_ids)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                logits = head(features)
            loss, bce, tversky = interior_wall_loss(
                logits.float(),
                target,
                footprint,
                taxonomy,
                target_dilation=dilation,
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in (("loss", loss), ("bce", bce), ("tversky", tversky)):
                totals[key] += float(value.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        head.eval()
        ious = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target, footprint, floor_ids = target.to(device), footprint.to(device), floor_ids.to(device)
                logits = head(features_for(backbone, target, footprint, floor_ids))
                ious.append(interior_wall_iou(logits, target, footprint, taxonomy))
        record = {
            "epoch": epoch,
            "target_dilation": dilation,
            **{key: value / max(1, seen) for key, value in totals.items()},
            "interior_wall_iou": sum(ious) / max(1, len(ious)),
        }
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
        if record["interior_wall_iou"] > best:
            best = record["interior_wall_iou"]
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
