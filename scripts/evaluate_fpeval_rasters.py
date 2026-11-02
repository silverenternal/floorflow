#!/usr/bin/env python3
"""Evaluate saved prediction rasters with fpeval and geometry support masks."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.geometry_cache import load_geometry_cache
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rasters", type=Path, required=True)
    parser.add_argument("--raster-key", default="prediction")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--reconcile-use-ip", action="store_true")
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _support_iou(predicted: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    left = (predicted > 0) & region
    right = (target > 0) & region
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def _process(
    path: str,
    geometry_root: str,
    raster_key: str,
    reconcile: bool,
    reconcile_use_ip: bool,
) -> dict:
    raster_path = Path(path)
    root = Path(geometry_root)
    taxonomy = FloorplanTaxonomy.read(root / "taxonomy.json")
    floor = Path(raster_path.stem.replace("__", "/"))
    floor_name = floor.name.replace("-", "_")
    geometry_path = root / "test" / floor.parent / f"{floor_name}.geometry.npz"
    with np.load(raster_path) as raster:
        prediction = raster[raster_key]
        target = raster["target"]
        footprint = raster["footprint"] > 0
        predicted_support = raster.get("room_support", np.zeros_like(target))
    geometry = load_geometry_cache(geometry_path)
    occlusion = geometry["occlusion_mask"] > 0
    result = fpeval_raster_metrics(
        prediction,
        target,
        taxonomy,
        tolerance_fractions=(0.015, 0.05),
        reconcile=reconcile,
        reconcile_use_ip=reconcile_use_ip,
    )
    return {
        "case": raster_path.stem,
        "tolerances": result["summary"],
        "prediction_validity": result["prediction_validity"],
        "double_walls": result["prediction_double_walls"],
        "support_iou_all": _support_iou(
            predicted_support,
            geometry["room_support"],
            footprint,
        ),
        "support_iou_visible": _support_iou(
            predicted_support,
            geometry["room_support"],
            footprint & ~occlusion,
        ),
        "support_iou_occluded": _support_iou(
            predicted_support,
            geometry["room_support"],
            footprint & occlusion,
        ),
    }


def _aggregate(rows: list[dict]) -> dict:
    metrics = {}
    for tolerance in ("0.015", "0.05"):
        keys = rows[0]["tolerances"][tolerance]
        metrics[tolerance] = {
            key: float(
                np.mean(
                    [
                        row["tolerances"][tolerance][key]
                        for row in rows
                        if isinstance(row["tolerances"][tolerance][key], (int, float))
                    ]
                )
            )
            for key in keys
            if isinstance(rows[0]["tolerances"][tolerance][key], (int, float))
        }
    metrics["validity_fraction"] = float(
        np.mean([row["prediction_validity"]["valid"] for row in rows])
    )
    metrics["watertight_fraction"] = float(
        np.mean([row["prediction_validity"]["watertight"] for row in rows])
    )
    metrics["double_walls"] = int(sum(row["double_walls"] for row in rows))
    for key in (
        "support_iou_all",
        "support_iou_visible",
        "support_iou_occluded",
    ):
        metrics[key] = float(np.mean([row[key] for row in rows]))
    return metrics


def main() -> None:
    args = parse_args()
    paths = sorted(args.rasters.glob("*.npz"))
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _process,
                str(path),
                str(args.geometry_root),
                args.raster_key,
                args.reconcile,
                args.reconcile_use_ip,
            )
            for path in paths
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0 or completed == len(paths):
                print(f"evaluated {completed}/{len(paths)}", flush=True)
    rows.sort(key=lambda row: row["case"])
    report = {
        "case_count": len(rows),
        "metrics": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
