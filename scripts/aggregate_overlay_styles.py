#!/usr/bin/env python3
"""Aggregate per-style deterministic overlay evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    styles = {}
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        overlays = report["room_metrics"]["overlay"]
        constraints = report.get("constraint_metrics", {})
        styles[str(report["style_id"])] = {
            "path": str(path),
            "room_macro_iou": float(overlays["room_macro_iou"]),
            "instance_geometric_iou": float(
                overlays["instance_geometric_iou"]
            ),
            "instance_room_f1": float(overlays["instance_room_f1"]),
            "instance_type_accuracy": float(
                overlays["instance_type_accuracy"]
            ),
            **constraints,
        }
    values = list(styles.values())
    room_coverages = [
        row["room_coverage"]
        for row in values
        if "room_coverage" in row
    ]
    ious = [row["room_macro_iou"] for row in values]
    report = {
        "styles": styles,
        "aggregate": {
            "quality_mean": float(np.mean(ious)),
            "quality_worst": float(np.min(ious)),
            "coverage_mean": float(np.mean(room_coverages)),
            "coverage_worst": float(np.min(room_coverages)),
            "room_macro_iou_mean": float(np.mean(ious)),
            "room_macro_iou_worst": float(np.min(ious)),
            "instance_room_f1_mean": float(
                np.mean([row["instance_room_f1"] for row in values])
            ),
            "instance_type_accuracy_mean": float(
                np.mean([row["instance_type_accuracy"] for row in values])
            ),
            "room_coverage_mean": float(
                np.mean(
                    [
                        row["room_coverage"]
                        for row in values
                        if "room_coverage" in row
                    ]
                )
            ),
            "outside_footprint_pixels_mean": float(
                np.mean(
                    [
                        row["outside_footprint_pixels"]
                        for row in values
                        if "outside_footprint_pixels" in row
                    ]
                )
            ),
        },
        "best_of_n_used": False,
        "style_selection": "user supplied; no target-derived style",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
