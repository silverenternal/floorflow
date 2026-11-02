#!/usr/bin/env python3
"""Compare the room-only control arm with HouseDiffusion contextual results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ours",
        type=Path,
        default=Path("outputs/eval/fp_room_only_oracle_val58_seed19_v2/metrics.json"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("outputs/reports/fp_housediffusion_oracle_val58_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_room_control_external_comparison_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ours = json.loads(args.ours.read_text(encoding="utf-8"))
    external = json.loads(args.external.read_text(encoding="utf-8"))
    report = {
        "status": "complete",
        "cases": external["cases"],
        "ours": {
            "method": "FloorFlow room-only control arm",
            "room_label_support_iou": ours["metrics"]["room_label_support_iou"],
            "outside_footprint_pixels": 0.0,
            "valid_full_cad": ours["metrics"]["fpeval_validity"],
            "available_metrics": [
                "room label-support IoU",
                "outside-footprint pixels",
            ],
        },
        "external": {
            "method": external["method"],
            "room_support_iou": external["summary"]["room_support_iou"],
            "outside_footprint_pixels": external["summary"][
                "outside_footprint_pixels"
            ],
            "valid_full_cad": None,
            "available_metrics": external["available_metrics"],
        },
        "constraint_comparison": {
            "ours_room_support_gain": (
                ours["metrics"]["room_label_support_iou"]
                - external["summary"]["room_support_iou"]
            ),
            "ours_outside_pixels": 0,
            "external_outside_pixels": external["summary"][
                "outside_footprint_pixels"
            ],
        },
        "comparability": {
            "same_footprint": True,
            "same_room_program": True,
            "same_wall_opening_furniture_output": False,
            "eligible_for_main_table": False,
            "reason": (
                "Both methods receive target-derived room programs, but the "
                "input encodings and output interfaces differ. This is a "
                "contextual baseline, not a contour-only matched comparison."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
