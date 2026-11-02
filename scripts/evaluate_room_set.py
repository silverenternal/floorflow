#!/usr/bin/env python3
"""Evaluate the query-based room-set head on a frozen split."""

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
from floorflow.heads.room_set import RoomSetHead, labels_to_room_targets, room_set_metrics
from floorflow.data.taxonomy import FloorplanTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/room_set/metrics.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
    config = checkpoint["config"]
    backbone, _, _, _ = load_checkpoint_model(config["checkpoint"], args.device, taxonomy=taxonomy)
    head = RoomSetHead(backbone.output.in_channels, taxonomy.num_classes).to(args.device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    dataset = CubiCasaFloorplanDataset(args.data_root, args.split, image_size=int(config.get("image_size", 256)), cache_root=args.cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=args.device)
            features = backbone._encode_features(zero, torch.zeros(target.shape[0], device=args.device), footprint, floor_ids=floor_ids.to(args.device))
            features = F.adaptive_avg_pool2d(features, (16, 16))
            classes, boxes, mask = labels_to_room_targets(target, taxonomy)
            metrics = room_set_metrics(head(features), classes, boxes, mask)
            rows.append(metrics)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary.update({"cases": len(dataset), "checkpoint": str(args.checkpoint)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
