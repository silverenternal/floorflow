#!/usr/bin/env python3
"""Compare official Raster2Seq room evidence with one FloorFlow readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster2seq", type=Path, required=True)
    parser.add_argument("--floorflow", type=Path, required=True)
    parser.add_argument("--floorflow-arm", default="blend50")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raster2seq = json.loads(args.raster2seq.read_text(encoding="utf-8"))
    floorflow = json.loads(args.floorflow.read_text(encoding="utf-8"))
    floorflow_metrics = floorflow["summary"][args.floorflow_arm]
    teacher = {
        key: float(raster2seq["summary"][key])
        for key in (
            "room_iou",
            "room_support_iou",
            "instance_f1",
            "instance_type_accuracy",
            "instance_count_abs_error",
        )
    }
    student = {
        "room_iou": float(floorflow_metrics["room_iou"]),
        "instance_f1": float(floorflow_metrics["instance_f1"]),
        "instance_type_accuracy": float(floorflow_metrics["type_accuracy"]),
        "instance_count_abs_error": float(floorflow_metrics["count_mae"]),
    }
    report = {
        "schema_version": "raster2seq-teacher-gap-v1",
        "cases": int(raster2seq["cases"]),
        "teacher": teacher,
        "student": {
            "arm": args.floorflow_arm,
            **student,
        },
        "teacher_minus_student": {
            key: teacher[key] - student[key]
            for key in student
        },
        "claim_boundary": (
            f"{int(raster2seq['cases'])} named validation cases with an "
            "official checkpoint; teacher eligibility evidence, not a "
            "full-split SOTA row."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

