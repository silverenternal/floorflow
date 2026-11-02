#!/usr/bin/env python3
"""Deterministic per-class routing between two semantic endpoint models."""

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
from floorflow.geometry.core import compose_known_exterior
from floorflow.inference.hybrid_decode import hybrid_decode
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.sampling_policy import SamplingPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--secondary-classes", nargs="+", choices=("furniture", "opening", "wall"), required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary, config, taxonomy, _ = load_checkpoint_model(args.primary, args.device)
    secondary, _secondary_config, secondary_taxonomy, _ = load_checkpoint_model(
        args.secondary,
        args.device,
        taxonomy=taxonomy,
    )
    if secondary_taxonomy.class_names != taxonomy.class_names:
        raise ValueError("primary/secondary taxonomy mismatch")
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, names in loader:
            target, footprint, floor_ids = target.to(args.device), footprint.to(args.device), floor_ids.to(args.device)
            primary_labels = semantic_endpoint_labels(primary, target, footprint, floor_ids)
            secondary_labels = semantic_endpoint_labels(secondary, target, footprint, floor_ids)
            hybrid_np = np.stack(
                [
                    hybrid_decode(
                        primary_labels[index].cpu().numpy(),
                        secondary_labels[index].cpu().numpy(),
                        taxonomy,
                        secondary_classes=tuple(args.secondary_classes),
                    )
                    for index in range(target.shape[0])
                ]
            )
            labels = compose_known_exterior(
                torch.from_numpy(hybrid_np).to(args.device),
                footprint,
                opening_ids=taxonomy.opening_ids,
            )
            for index, name in enumerate(names):
                rows.append(
                    {
                        "case": name,
                        **floorplan_metrics(
                            labels[index : index + 1],
                            target[index : index + 1],
                            footprint[index : index + 1],
                            taxonomy,
                        ),
                    }
                )
    keys = (
        "room_macro_iou",
        "room_area_coverage",
        "room_component_gap",
        "opening_f1",
        "opening_wall_alignment",
        "furniture_iou",
        "furniture_room_containment",
        "interior_wall_iou",
        "wall_centerline_f1",
        "wall_closure",
    )
    report = {
        "cases": len(rows),
        "claim_role": "deterministic_class_group_routing",
        "best_of_n_used": False,
        "sampling_policy": SamplingPolicy(purpose="single_sample").manifest(),
        "primary": str(args.primary),
        "secondary": str(args.secondary),
        "secondary_classes": args.secondary_classes,
        "metrics": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
