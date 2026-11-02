#!/usr/bin/env python3
"""Tune object-head decision thresholds on validation data only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.heads.structured_objects import StructuredObjectHeads, object_group_metrics, object_group_predictions
from floorflow.data.taxonomy import FloorplanTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/structured/threshold_sweep.json")
    args = parser.parse_args()
    payload = torch.load(args.object_checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    backbone, config, _, _ = load_checkpoint_model(args.semantic_checkpoint, args.device, taxonomy=taxonomy)
    head = StructuredObjectHeads(backbone.output.in_channels, taxonomy).to(args.device)
    head.load_state_dict(payload["head"])
    head.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    cached = []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target, footprint, floor_ids = target.to(args.device), footprint.to(args.device), floor_ids.to(args.device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=args.device)
            outputs = head(
                backbone._encode_features(
                    zero,
                    torch.zeros(target.shape[0], device=args.device),
                    footprint,
                    floor_ids=floor_ids,
                )
            )
            cached.append((outputs, target))
    opening_rows = []
    furniture_rows = []
    for threshold in args.thresholds:
        opening_metrics = []
        furniture_metrics = []
        for outputs, target in cached:
            opening_prediction = object_group_predictions(
                outputs,
                taxonomy,
                opening_threshold=threshold,
                furniture_threshold=0.0,
            )
            furniture_prediction = object_group_predictions(
                outputs,
                taxonomy,
                opening_threshold=0.0,
                furniture_threshold=threshold,
            )
            opening_metrics.append(object_group_metrics(opening_prediction, target, taxonomy))
            furniture_metrics.append(object_group_metrics(furniture_prediction, target, taxonomy))
        opening_rows.append(
            {
                "threshold": threshold,
                **{key: float(np.mean([row[key] for row in opening_metrics])) for key in opening_metrics[0]},
            }
        )
        furniture_rows.append(
            {
                "threshold": threshold,
                **{key: float(np.mean([row[key] for row in furniture_metrics])) for key in furniture_metrics[0]},
            }
        )
    report = {
        "claim_role": "validation_threshold_tuning",
        "split": args.split,
        "best_opening": max(opening_rows, key=lambda row: row["opening_f1"]),
        "best_furniture": max(furniture_rows, key=lambda row: row["furniture_f1"]),
        "opening_rows": opening_rows,
        "furniture_rows": furniture_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
