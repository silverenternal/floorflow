#!/usr/bin/env python3
"""Measure room-label support replication from frozen test rasters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="SEED=RASTER_DIRECTORY",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_room_label_support_test503_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    room_ids = np.asarray(taxonomy.room_ids)
    rows = []
    for value in args.run:
        seed_text, directory_text = value.split("=", 1)
        values = []
        for path in sorted(Path(directory_text).glob("*.npz")):
            with np.load(path) as payload:
                predicted = np.isin(payload["room_labels"], room_ids)
                target = np.isin(payload["target"], room_ids)
                footprint = payload["footprint"].astype(bool)
            predicted &= footprint
            target &= footprint
            union = np.logical_or(predicted, target).sum()
            values.append(
                1.0
                if union == 0
                else float(np.logical_and(predicted, target).sum() / union)
            )
        values = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "seed": int(seed_text),
                "raster_directory": directory_text,
                "cases": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "min": float(values.min()),
            }
        )
    report = {
        "status": "complete",
        "metric": "room label-support IoU",
        "scope": "CubiCasa test503, saved visible room labels",
        "seeds": rows,
        "checks": {
            "all_cases_503": all(row["cases"] == 503 for row in rows),
            "all_mean_at_least_0_95": all(
                row["mean"] >= 0.95 for row in rows
            ),
            "max_minus_min_mean_le_0_02": (
                max(row["mean"] for row in rows)
                - min(row["mean"] for row in rows)
            )
            <= 0.02,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
