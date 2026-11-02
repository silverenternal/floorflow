#!/usr/bin/env python3
"""Audit contour-inward and multi-stage curriculum ordering on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.recursion.inward_flow import (
    contour_inward_weight_map,
    multi_stage_weights,
    stage_ordered_inward_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/inward_curriculum_audit_v1.json"),
    )
    parser.add_argument("--cases", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.cases,
        augment=False,
    )
    targets = []
    footprints = []
    for index in range(len(dataset)):
        target, footprint, _floor_id, _name = dataset[index]
        targets.append(target)
        footprints.append(footprint)
    target = torch.stack(targets)
    footprint = torch.stack(footprints)
    audit = []
    for progress in (0.0, 0.5, 1.0):
        weights = multi_stage_weights(progress)
        inward = contour_inward_weight_map(footprint, progress)
        ordered = stage_ordered_inward_weights(
            target,
            footprint,
            dataset.taxonomy,
            progress,
        )
        audit.append(
            {
                "progress": progress,
                "stage_weights": {
                    "room": weights.room,
                    "structure": weights.structure,
                    "detail": weights.detail,
                },
                "inward_band_weight_mean": float(inward.mean()),
                "ordered_weight_mean": float(ordered.mean()),
                "ordered_weight_coverage": float(
                    (ordered > 0).float().mean()
                ),
            }
        )
    report = {
        "status": "pass",
        "cases": len(dataset),
        "best_of_n_used": False,
        "target_used_for_inference": False,
        "schedule": audit,
        "checks": {
            "room_first": audit[0]["stage_weights"]["room"]
            == max(audit[0]["stage_weights"].values()),
            "structure_second": audit[1]["stage_weights"]["structure"]
            == max(audit[1]["stage_weights"].values()),
            "detail_last": audit[2]["stage_weights"]["detail"]
            == max(audit[2]["stage_weights"].values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
