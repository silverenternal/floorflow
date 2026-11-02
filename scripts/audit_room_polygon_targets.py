#!/usr/bin/env python3
"""Audit that extracted room loops round-trip to the original room raster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.evaluation.conditional_ceiling import room_macro_iou_numpy
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.polygon_targets import (
    polygon_instance_targets,
    rasterize_room_polygons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--max-instances", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit,
        write_cache=False,
    )
    rows = []
    for index in range(len(dataset)):
        target, _footprint, _floor_id, name = dataset[index]
        target = target[None]
        classes, polygons, valid = polygon_instance_targets(
            target,
            dataset.taxonomy,
            max_instances=args.max_instances,
            points=args.points,
        )
        reconstructed = rasterize_room_polygons(
            polygons,
            classes,
            valid,
            dataset.taxonomy,
            height=target.shape[-2],
            width=target.shape[-1],
        )
        target_np = target[0].numpy()
        reconstructed_np = reconstructed[0].cpu().numpy()
        room_ids = np.asarray(dataset.taxonomy.room_ids)
        target_rooms = np.isin(target_np, room_ids)
        reconstructed_rooms = np.isin(reconstructed_np, room_ids)
        union = np.logical_or(target_rooms, reconstructed_rooms).sum()
        rows.append(
            {
                "name": name,
                "room_macro_iou": room_macro_iou_numpy(
                    reconstructed_np,
                    target_np,
                    dataset.taxonomy,
                ),
                "room_occupancy_iou": float(
                    np.logical_and(target_rooms, reconstructed_rooms).sum()
                    / max(1, union)
                ),
                "instances": int(valid.sum()),
            }
        )
    report = {
        "split": args.split,
        "case_count": len(rows),
        "points": args.points,
        "max_instances": args.max_instances,
        "room_macro_iou": float(np.mean([row["room_macro_iou"] for row in rows])),
        "room_occupancy_iou": float(
            np.mean([row["room_occupancy_iou"] for row in rows])
        ),
        "max_instance_saturation_fraction": float(
            np.mean([row["instances"] >= args.max_instances for row in rows])
        ),
        "worst_cases": sorted(rows, key=lambda row: row["room_macro_iou"])[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "room_macro_iou", "room_occupancy_iou")}, indent=2))


if __name__ == "__main__":
    main()
