#!/usr/bin/env python3
"""Audit geometry-cache coverage, raster agreement, and occlusion structure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.geometry_cache import (
    PackedPolygons,
    load_geometry_cache,
    rasterize_room_polygons,
)
from floorflow.evaluation.conditional_ceiling import room_macro_iou_numpy
from floorflow.evaluation.room_metric_suite import matched_instance_metrics
from floorflow.data.taxonomy import FloorplanTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/geometry_cache_test_audit.json",
    )
    return parser.parse_args()


def _visible_room_recall(
    support: np.ndarray,
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    visible = np.isin(target, np.asarray(taxonomy.room_ids))
    if not visible.any():
        return 1.0
    return float(np.logical_and(support > 0, visible).sum() / visible.sum())


def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.geometry_root / "taxonomy.json")
    geometry_files = sorted((args.geometry_root / args.split).rglob("*.geometry.npz"))
    rows = []
    for path in geometry_files:
        relative = path.relative_to(args.geometry_root / args.split)
        raster_path = args.raster_root / args.split / relative.with_name(
            relative.name.replace(".geometry.npz", ".npz")
        )
        geometry = load_geometry_cache(path)
        with np.load(raster_path) as raster:
            raster_target = raster["target"]
            raster_footprint = raster["footprint"]
        rooms = PackedPolygons(
            points=geometry["room_points"],
            offsets=geometry["room_offsets"],
        )
        roundtrip = rasterize_room_polygons(
            rooms,
            geometry["room_class_ids"],
            int(geometry["target"].shape[0]),
        )
        roundtrip_visible = np.where(
            geometry["occlusion_mask"] > 0,
            0,
            roundtrip,
        )
        instances = matched_instance_metrics(
            roundtrip_visible,
            raster_target,
            taxonomy,
        )
        host_distance = geometry["opening_host_distances"]
        valid_distance = host_distance[host_distance >= 0]
        rows.append(
            {
                "case": str(relative),
                "target_equal": bool(np.array_equal(geometry["target"], raster_target)),
                "legacy_target_mismatch_fraction": float(
                    np.mean(geometry["target"] != raster_target)
                ),
                "footprint_equal": bool(
                    np.array_equal(geometry["footprint"], raster_footprint)
                ),
                "room_polygons": int(len(geometry["room_offsets"]) - 1),
                "wall_polygons": int(len(geometry["wall_offsets"]) - 1),
                "opening_polygons": int(len(geometry["opening_offsets"]) - 1),
                "furniture_polygons": int(len(geometry["furniture_offsets"]) - 1),
                "visible_room_recall": _visible_room_recall(
                    geometry["room_support"],
                    raster_target,
                    taxonomy,
                ),
                "occlusion_pixels": int(geometry["occlusion_mask"].sum()),
                "unattached_openings": int(
                    (geometry["opening_wall_indices"] < 0).sum()
                ),
                "opening_host_distance_p50": (
                    float(np.quantile(valid_distance, 0.5))
                    if valid_distance.size
                    else None
                ),
                "opening_host_distance_p95": (
                    float(np.quantile(valid_distance, 0.95))
                    if valid_distance.size
                    else None
                ),
                "roundtrip_instance_iou": instances["matched_geometric_iou"],
                "roundtrip_instance_f1": instances["instance_f1_at_threshold"],
                "roundtrip_semantic_iou": room_macro_iou_numpy(
                    roundtrip_visible,
                    raster_target,
                    taxonomy,
                ),
            }
        )
    report = {
        "split": args.split,
        "case_count": len(rows),
        "target_equal_fraction": float(np.mean([row["target_equal"] for row in rows])),
        "legacy_target_mismatch_fraction_mean": float(
            np.mean([row["legacy_target_mismatch_fraction"] for row in rows])
        ),
        "footprint_equal_fraction": float(
            np.mean([row["footprint_equal"] for row in rows])
        ),
        "visible_room_recall_min": float(
            min((row["visible_room_recall"] for row in rows), default=0.0)
        ),
        "visible_room_recall_mean": float(
            np.mean([row["visible_room_recall"] for row in rows])
        ),
        "roundtrip_instance_iou_mean": float(
            np.mean([row["roundtrip_instance_iou"] for row in rows])
        ),
        "roundtrip_instance_f1_mean": float(
            np.mean([row["roundtrip_instance_f1"] for row in rows])
        ),
        "roundtrip_semantic_iou_mean": float(
            np.mean([row["roundtrip_semantic_iou"] for row in rows])
        ),
        "unattached_openings": int(
            sum(row["unattached_openings"] for row in rows)
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in report if key != "rows"},
            indent=2,
        )
    )
    if not (
        report["footprint_equal_fraction"] == 1.0
        and report["roundtrip_instance_iou_mean"] >= 0.95
        and report["roundtrip_semantic_iou_mean"] >= 0.95
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
