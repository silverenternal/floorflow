#!/usr/bin/env python3
"""Aggregate seed-specific dense/query evidence without hiding instability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.dense_query_blend_multiseed import (
    aggregate_split,
    load_split_reports,
    promotion_gates,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ("val491", "cal64", "holdout128", "structure_val491")
CANDIDATE_METRICS = (
    "room_iou",
    "instance_f1",
    "instance_precision",
    "instance_recall",
    "type_accuracy",
    "count_mae",
    "room_support_iou",
)
STRUCTURE_METRICS = (
    "room_macro_iou",
    "room_support_iou",
    "instance_instance_f1_at_threshold",
    "instance_count_abs_error",
    "instance_matched_type_accuracy",
    "fpeval_wall_f1",
    "fpeval_opening_f1",
    "node_f1",
    "edge_overlap",
    "ged",
    "validity",
    "functional_watertight",
    "wall_continuity_rate",
    "outside_footprint_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--report-root", type=Path, default=ROOT / "outputs/reports")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/dense_query_blend_fiveseed_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths_by_split = {
        split: [
            args.report_root
            / f"dense_query_blend_seed{seed}_{split}_v1.json"
            for seed in args.seeds
        ]
        for split in args.splits
    }
    missing = {
        split: [str(path) for path in paths if not path.is_file()]
        for split, paths in paths_by_split.items()
    }
    missing = {split: paths for split, paths in missing.items() if paths}
    splits = {}
    for split, paths in paths_by_split.items():
        existing = [path for path in paths if path.is_file()]
        if not existing:
            continue
        reports = load_split_reports(existing)
        metric_keys = (
            STRUCTURE_METRICS if split.startswith("structure_") else CANDIDATE_METRICS
        )
        splits[split] = aggregate_split(reports, metric_keys=metric_keys)
    gates = promotion_gates(splits, expected_seeds=args.seeds)
    report = {
        "schema_version": "dense-query-blend-fiveseed-v1",
        "protocol": "predicted_control",
        "test_opened": False,
        "expected_seeds": args.seeds,
        "missing_files": missing,
        "splits": splits,
        "gates": gates,
        "decision": "promote" if gates["passed"] else "hold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
