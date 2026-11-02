#!/usr/bin/env python3
"""Aggregate the frozen five-seed predicted-control plausibility reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "room_macro_iou_diagnostic",
    "room_coverage",
    "raster_plausible",
    "wall_area_fraction",
    "componentwise_wall_closure",
    "opening_wall_alignment",
    "furniture_room_containment",
    "outside_footprint_fraction",
    "room_face_retention",
    "validity",
    "watertight",
    "functional_watertight",
    "room_overlap_ratio",
    "wall_continuity_rate",
    "plan_plausible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        nargs=5,
        type=Path,
        default=(
            ROOT / "outputs/reports/predicted_control_plausibility_val491_final_v12.json",
            ROOT / "outputs/reports/predicted_control_plausibility_seed20_v12.json",
            ROOT / "outputs/reports/predicted_control_plausibility_seed21_v12.json",
            ROOT / "outputs/reports/predicted_control_plausibility_seed22_v12.json",
            ROOT / "outputs/reports/predicted_control_plausibility_seed23_v12.json",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_plausibility_fiveseed_v12.json",
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(len(array))
    half_width = float(student_t.ppf(0.975, len(array) - 1)) * standard_error
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "range": float(array.max() - array.min()),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    args = parse_args()
    seeds = (19, 20, 21, 22, 23)
    rows = {}
    for seed, path in zip(seeds, args.reports, strict=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["arms"]["union0_explicit"]["metrics"]
        rows[str(seed)] = {
            "report": str(path),
            "metrics": {
                name: float(metrics[name])
                for name in METRICS
            },
        }
    summary = {
        name: summarize(
            [row["metrics"][name] for row in rows.values()]
        )
        for name in METRICS
    }
    gate_by_seed = {
        seed: {
            "raster_plausibility": row["metrics"]["room_coverage"] >= 0.95
            and row["metrics"]["componentwise_wall_closure"] >= 0.95
            and row["metrics"]["outside_footprint_fraction"] == 0.0,
            "cad_identity": row["metrics"]["room_face_retention"] >= 0.95,
            "functional_topology": row["metrics"]["validity"] == 1.0
            and row["metrics"]["functional_watertight"] >= 0.95
            and row["metrics"]["room_overlap_ratio"] == 0.0
            and row["metrics"]["wall_continuity_rate"] >= 0.95,
            "plan_plausibility": row["metrics"]["plan_plausible"] >= 0.95,
        }
        for seed, row in rows.items()
    }
    report = {
        "schema_version": "predicted-control-plausibility-fiveseed-v1",
        "scope": "Frozen configuration, all validation cases, no test access",
        "configuration": {
            "room_readout": "explicit_partition_polygons",
            "wall_readout": "union radius 0 with exterior composition",
            "gap_reconcile_fraction": 0.04,
            "room_checkpoint": (
                "outputs/train/raster_room_control_native_full_v1/best.pt"
            ),
            "structure_checkpoints": [
                "outputs/train/raster_structure_bridge_full_v1/best.pt",
                "outputs/train/raster_structure_bridge_full_seed20_v1/best.pt",
                "outputs/train/raster_structure_bridge_full_seed21_v1/best.pt",
                "outputs/train/raster_structure_bridge_full_seed22_v1/best.pt",
                "outputs/train/raster_structure_bridge_full_seed23_v1/best.pt",
            ],
        },
        "validation_cases_per_seed": 491,
        "test_opened": False,
        "target_used_at_inference": False,
        "ci_method": "student-t interval, two-sided 95%, n=5",
        "seeds": rows,
        "summary": summary,
        "gate_by_seed": gate_by_seed,
        "all_seeds_pass": all(
            all(gates.values()) for gates in gate_by_seed.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "all_seeds_pass": report["all_seeds_pass"]}, indent=2))


if __name__ == "__main__":
    main()
