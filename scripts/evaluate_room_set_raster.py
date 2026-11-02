#!/usr/bin/env python3
"""Render room-set predictions and compare them with semantic room maps."""

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
from floorflow.geometry.core import compose_known_exterior
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.heads.room_set import RoomSetHead, compose_room_set_with_semantics, render_room_set
from floorflow.data.taxonomy import FloorplanTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-set-checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/room_set/raster_metrics.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.room_set_checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
    backbone, config, _, _ = load_checkpoint_model(args.semantic_checkpoint, args.device, taxonomy=taxonomy)
    head = RoomSetHead(backbone.output.in_channels, taxonomy.num_classes).to(args.device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    dataset = CubiCasaFloorplanDataset(args.data_root, "test", image_size=int(config.get("image_size", 256)), cache_root=args.cache_root, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    direct_rows, set_rows = [], []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target = target.to(args.device)
            footprint = footprint.to(args.device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=args.device)
            features = backbone._encode_features(zero, torch.zeros(target.shape[0], device=args.device), footprint, floor_ids=floor_ids.to(args.device))
            semantic_logits = backbone.semantic_endpoint(features)
            semantic_scores = torch.cat([torch.zeros_like(semantic_logits[:, :1]), semantic_logits], dim=1)
            semantic_labels = semantic_scores.argmax(dim=1)
            room_predictions = head(F.adaptive_avg_pool2d(features, (16, 16)))
            room_labels = render_room_set(room_predictions, target.shape[-2], target.shape[-1])
            combined = compose_room_set_with_semantics(semantic_labels, room_labels, footprint, taxonomy)
            composite = compose_known_exterior(combined, footprint, opening_ids=taxonomy.opening_ids)
            for index in range(target.shape[0]):
                direct_rows.append(floorplan_metrics(semantic_labels[index : index + 1], target[index : index + 1], footprint[index : index + 1], taxonomy))
                set_rows.append(floorplan_metrics(composite[index : index + 1], target[index : index + 1], footprint[index : index + 1], taxonomy))
    keys = ("room_macro_iou", "room_area_coverage", "furniture_iou", "opening_f1", "interior_wall_iou")
    report = {
        "cases": len(dataset),
        "direct": {key: float(np.mean([row[key] for row in direct_rows])) for key in keys},
        "room_set_composite": {key: float(np.mean([row[key] for row in set_rows])) for key in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
