#!/usr/bin/env python3
"""Measure the train-only nearest-contour room prior as a non-generative baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.core import contour_descriptor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/retrieval_prior/report.json")
    args = parser.parse_args()
    source = CubiCasaFloorplanDataset(
        args.data_root,
        args.source_split,
        image_size=256,
        cache_root=args.cache_root,
        preload=True,
    )
    target = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        preload=True,
    )
    taxonomy = source.taxonomy
    descriptors = np.stack([contour_descriptor(item) for item in source._footprints]).astype(np.float64)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True).clip(min=1.0e-8)
    rows = []
    for index in range(len(target)):
        query = contour_descriptor(target._footprints[index]).astype(np.float64)
        query /= max(np.linalg.norm(query), 1.0e-8)
        nearest = int(np.argmax(descriptors @ query))
        prediction = source._targets[nearest]
        expected = target._targets[index]
        ious = []
        for class_id in taxonomy.room_ids:
            mask = expected == class_id
            if mask.any():
                predicted = prediction == class_id
                ious.append(float((mask & predicted).sum() / max(1, (mask | predicted).sum())))
        room_prediction = np.isin(prediction, taxonomy.room_ids)
        footprint = target._footprints[index].astype(bool)
        rows.append(
            {
                "case": target.samples[index].relative_case.as_posix(),
                "source_case": source.samples[nearest].relative_case.as_posix(),
                "room_macro_iou": float(np.mean(ious)) if ious else 0.0,
                "room_area_coverage": float((room_prediction & footprint).sum() / max(1, footprint.sum())),
            }
        )
    report = {
        "claim_role": "retrieval_baseline_and_coarse_prior",
        "best_of_n_used": False,
        "selection": "single_nearest_neighbour",
        "source_split": args.source_split,
        "target_split": args.split,
        "cases": len(rows),
        "room_macro_iou": float(np.mean([row["room_macro_iou"] for row in rows])),
        "room_macro_iou_median": float(np.median([row["room_macro_iou"] for row in rows])),
        "room_area_coverage": float(np.mean([row["room_area_coverage"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "room_macro_iou", "room_area_coverage")}, indent=2))


if __name__ == "__main__":
    main()
