#!/usr/bin/env python3
"""Train a query-based room instance head on a frozen contour backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.room_set import RoomSetHead, labels_to_room_targets, room_set_loss, room_set_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train/room_set")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    backbone, config, taxonomy, _ = load_checkpoint_model(args.checkpoint, device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = RoomSetHead(backbone.output.in_channels, taxonomy.num_classes).to(device)
    train = CubiCasaFloorplanDataset(args.data_root, "train", image_size=int(config.get("image_size", 256)), cache_root=args.cache_root, augment=True, limit=args.limit_train)
    val = CubiCasaFloorplanDataset(args.data_root, "val", image_size=int(config.get("image_size", 256)), cache_root=args.cache_root, augment=False, limit=args.limit_val)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        head.train()
        total = 0.0
        seen = 0
        for target, footprint, floor_ids, _names in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            floor_ids = floor_ids.to(device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=device)
            with torch.no_grad():
                features = backbone._encode_features(zero, torch.zeros(target.shape[0], device=device), footprint, floor_ids=floor_ids)
                features = F.adaptive_avg_pool2d(features, (16, 16))
            predictions = head(features)
            target_classes, target_boxes, target_mask = labels_to_room_targets(target, taxonomy)
            loss = room_set_loss(predictions, target_classes, target_boxes, target_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * target.shape[0]
            seen += target.shape[0]
        head.eval()
        rows = []
        with torch.no_grad():
            for target, footprint, floor_ids, _names in val_loader:
                target = target.to(device)
                footprint = footprint.to(device)
                floor_ids = floor_ids.to(device)
                zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=device)
                features = F.adaptive_avg_pool2d(
                    backbone._encode_features(zero, torch.zeros(target.shape[0], device=device), footprint, floor_ids=floor_ids),
                    (16, 16),
                )
                target_classes, target_boxes, target_mask = labels_to_room_targets(target, taxonomy)
                rows.append(room_set_metrics(head(features), target_classes, target_boxes, target_mask))
        val_loss = total / max(1, seen)
        record = {
            "epoch": epoch,
            "train_loss": val_loss,
            **{key: sum(row[key] for row in rows) / max(1, len(rows)) for key in rows[0]},
        }
        print(json.dumps(record), flush=True)
        (args.output_dir / "train.jsonl").open("a", encoding="utf-8").write(json.dumps(record) + "\n")
        checkpoint = {"head": head.state_dict(), "backbone": backbone.state_dict(), "taxonomy": taxonomy.to_json(), "config": vars(args)}
        torch.save(checkpoint, args.output_dir / "last.pt")
        quality = -record["matched_box_iou"] + record["train_loss"]
        if quality < best:
            best = quality
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
