#!/usr/bin/env python3
"""Build paired calibration/holdout statistics for room crop candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta


def _rows(path: Path, blend: float) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["case_rows"][str(blend)]


def _comparison(
    candidate: list[dict],
    baseline: list[dict],
    blend: float,
) -> dict:
    metrics = (
        "room_macro_iou",
        "instance_room_f1",
        "instance_type_accuracy",
        "instance_count_abs_error",
    )
    return {
        metric: paired_delta(
            candidate,
            baseline,
            metric,
            iterations=5000,
            seed=19,
            lower_is_better=metric == "instance_count_abs_error",
        )
        for metric in metrics
    } | {"blend": blend}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-cal", type=Path, required=True)
    parser.add_argument("--candidate-cal", type=Path, required=True)
    parser.add_argument("--baseline-holdout", type=Path, required=True)
    parser.add_argument("--candidate-holdout", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "room-pagefloor-paired-report-v1",
        "blend": args.blend,
        "calibration": _comparison(
            _rows(args.candidate_cal, args.blend),
            _rows(args.baseline_cal, args.blend),
            args.blend,
        ),
        "holdout": _comparison(
            _rows(args.candidate_holdout, args.blend),
            _rows(args.baseline_holdout, args.blend),
            args.blend,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
