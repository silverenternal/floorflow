#!/usr/bin/env python3
"""Aggregate supplemental structural-legality metrics across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    "invalid_rate",
    "parsable",
    "room_closure_rate",
    "wall_continuity_rate",
    "self_intersection_rate",
    "room_overlap_ratio",
    "out_of_boundary_ratio",
    "node_f1",
    "edge_overlap",
    "ged",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-report",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.seed_report
    ]
    summary = {
        metric: {
            "mean": float(
                np.mean([report["aggregate"][metric] for report in reports])
            ),
            "min": float(
                min(report["aggregate"][metric] for report in reports)
            ),
            "max": float(
                max(report["aggregate"][metric] for report in reports)
            ),
        }
        for metric in METRICS
    }
    report = {
        "status": "complete",
        "seeds": [int(report["seed"]) for report in reports],
        "cases_per_seed": [int(report["cases"]) for report in reports],
        "summary": summary,
        "checks": {
            "invalid_rate_zero": summary["invalid_rate"]["max"] == 0.0,
            "parsable_rate_one": summary["parsable"]["min"] == 1.0,
            "room_closure_rate_one": (
                summary["room_closure_rate"]["min"] == 1.0
            ),
            "self_intersection_rate_zero": (
                summary["self_intersection_rate"]["max"] == 0.0
            ),
            "out_of_boundary_ratio_zero": (
                summary["out_of_boundary_ratio"]["max"] == 0.0
            ),
            "room_overlap_ratio_below_0_01": (
                summary["room_overlap_ratio"]["max"] < 0.01
            ),
            "wall_continuity_at_least_0_94": (
                summary["wall_continuity_rate"]["min"] >= 0.94
            ),
        },
        "interpretation": (
            "Raster/CAD legality checks are stable, while room-graph identity "
            "metrics remain substantially below exact reconstruction and are "
            "reported only as appendix diagnostics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Structural Legality Audit",
        "",
        "| Metric | Mean | Range |",
        "|---|---:|---:|",
    ]
    labels = {
        "invalid_rate": "Invalid rate",
        "parsable": "Parsable rate",
        "room_closure_rate": "Room closure rate",
        "wall_continuity_rate": "Wall continuity rate",
        "self_intersection_rate": "Self-intersection rate",
        "room_overlap_ratio": "Room overlap ratio",
        "out_of_boundary_ratio": "Out-of-boundary ratio",
        "node_f1": "Node F1",
        "edge_overlap": "Edge overlap",
        "ged": "GED",
    }
    for metric in METRICS:
        values = summary[metric]
        table.append(
            f"| {labels[metric]} | {values['mean']:.6f} | "
            f"{values['min']:.6f}-{values['max']:.6f} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
