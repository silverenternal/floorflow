#!/usr/bin/env python3
"""Add polygon-room fpeval metrics to existing frozen raster evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    rows = json.loads(
        (args.source_dir / "per_case.json").read_text(encoding="utf-8")
    )
    raster_paths = sorted((args.source_dir / "rasters").glob("*.npz"))
    if len(rows) != len(raster_paths):
        raise ValueError("per-case rows and raster count do not match")
    for row, path in zip(rows, raster_paths):
        with np.load(path) as raster:
            report = fpeval_raster_metrics(
                raster["prediction"],
                raster["target"],
                taxonomy,
                tolerance_fractions=(0.05,),
                explicit_prediction_rooms=True,
                explicit_target_rooms=True,
            )
        polygon = report["summary"]["0.05"]
        row["fpeval_room_polygon_f1"] = polygon["room_f1"]
        row["fpeval_room_polygon_precision"] = polygon["room_precision"]
        row["fpeval_room_polygon_recall"] = polygon["room_recall"]
        if "fpeval_room_face_f1" not in row:
            row["fpeval_room_face_f1"] = row.get("fpeval_room_f1", 0.0)
        print(f"evaluated {len([r for r in rows if 'fpeval_room_polygon_f1' in r])}/{len(rows)}", flush=True)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    conditional_watertight = [
        row["fpeval_watertight"]
        for row in rows
        if row["target_watertight"] > 0.5
    ]
    metrics["conditional_watertight"] = float(np.mean(conditional_watertight))
    report = {
        "cases": len(rows),
        "task_variant": "contour + room program + coarse room/structure/furniture layout",
        "best_of_n_used": False,
        "single_deterministic_sample": True,
        "metrics": metrics,
        "target_watertight_fraction": float(
            np.mean([row["target_watertight"] > 0.5 for row in rows])
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_case.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
