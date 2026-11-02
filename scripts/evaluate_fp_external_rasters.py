#!/usr/bin/env python3
"""Evaluate external floorplan rasters on the hard metrics they support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.external_baseline_metrics import (
    HOUSE_ROOM_COLORS,
    HOUSE_TYPE_TO_CUBICASA,
    _read_rgb,
    _rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--stem-suffix", default="c_pred")
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="white",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    class_ids = {
        name: int(index)
        for index, name in enumerate(dataset.taxonomy.class_names)
    }
    room_type_map = {}
    for house_type, names in HOUSE_TYPE_TO_CUBICASA.items():
        available = [class_ids[name] for name in names if name in class_ids]
        if available:
            room_type_map[house_type] = available[0]
    rows = []
    for index in range(len(dataset)):
        target, footprint, _floor_id, name = dataset[index]
        target = np.asarray(target)
        footprint = np.asarray(footprint).astype(bool)
        image = _read_rgb(
            args.image_dir / f"{index}{args.stem_suffix}.png",
            target.shape,
        )
        generated = image.min(axis=-1) < 250
        if args.background == "black":
            generated &= image.max(axis=-1) > 0
        labels = np.zeros_like(target, dtype=np.uint8)
        for house_type, class_id in room_type_map.items():
            color = np.asarray(_rgb(HOUSE_ROOM_COLORS[house_type]))
            mask = np.linalg.norm(
                image.astype(np.int16) - color,
                axis=-1,
            ) <= 3
            labels[mask] = class_id
        room_support = labels > 0
        target_support = np.isin(target, dataset.taxonomy.room_ids)
        support_union = np.logical_or(room_support, target_support).sum()
        room_components = 0
        for class_id in np.unique(labels):
            if class_id == 0:
                continue
            _components, count = ndimage.label(labels == class_id)
            room_components += int(count)
        rows.append(
            {
                "id": str(name),
                "generated_support_pixels": int(generated.sum()),
                "outside_footprint_pixels": int(
                    np.logical_and(generated, ~footprint).sum()
                ),
                "footprint_coverage": float(
                    np.logical_and(generated, footprint).sum()
                    / max(1, footprint.sum())
                ),
                "room_support_iou": float(
                    np.logical_and(room_support, target_support).sum()
                    / max(1, support_union)
                ),
                "room_components": room_components,
            }
        )
    report = {
        "status": "complete",
        "method": args.method,
        "condition_mode": "oracle_room_program",
        "target_derived_control": True,
        "best_of_n_used": False,
        "cases": len(rows),
        "available_metrics": [
            "outside_footprint_pixels",
            "footprint_coverage",
            "room_support_iou",
            "room_components",
        ],
        "unavailable_metrics": [
            "wall F1",
            "opening F1",
            "watertightness",
            "furniture metrics",
            "fpeval edit cost",
        ],
        "summary": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "outside_footprint_pixels",
                "footprint_coverage",
                "room_support_iou",
                "room_components",
            )
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
