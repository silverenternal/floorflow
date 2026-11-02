#!/usr/bin/env python3
"""Compare matched dense/query furniture reports and write decision evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.cad_furniture_ablation import (
    compare_furniture_arm_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense",
        type=Path,
        default=ROOT / "outputs/train/cad_furniture_dense_fast64_v1/report.json",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=ROOT / "outputs/train/cad_furniture_query_fast64_v1/report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_fast64_comparison_v1.json",
    )
    parser.add_argument(
        "--dense-eval",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_dense_fast64_eval_v1.json",
    )
    parser.add_argument(
        "--query-eval",
        type=Path,
        default=ROOT / "outputs/reports/cad_furniture_query_fast64_eval_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense = json.loads(args.dense.read_text(encoding="utf-8"))
    query = json.loads(args.query.read_text(encoding="utf-8"))
    dense_eval = json.loads(args.dense_eval.read_text(encoding="utf-8"))
    query_eval = json.loads(args.query_eval.read_text(encoding="utf-8"))
    dense_summary = dense_eval["summary"]
    query_summary = query_eval["summary"]
    report = compare_furniture_arm_reports(dense, query)
    eval_keys = (
        "furniture_f1",
        "furniture_vector_f1",
        "furniture_vector_iou",
        "furniture_vector_count",
        "furniture_vector_precision",
        "furniture_vector_recall",
        "furniture_vector_polygon_valid_rate",
        "furniture_vector_label_accuracy",
        "furniture_vector_attachment_accuracy",
        "furniture_vector_containment",
        "furniture_vector_out_of_boundary",
        "room_iou",
        "wall_f1",
        "opening_f1",
        "vector_schema_validity",
        "vector_conditional_watertight",
    )
    report["selected_checkpoint_eval"] = {
        key: {
            "dense": float(dense_summary[key]),
            "query": float(query_summary[key]),
            "query_minus_dense": (
                float(query_summary[key]) - float(dense_summary[key])
            ),
        }
        for key in eval_keys
    }
    report["gates"]["vector_furniture_pass"] = (
        report["selected_checkpoint_eval"]["furniture_vector_f1"][
            "query_minus_dense"
        ]
        > 0.0
    )
    report["gates"]["promotion_ready"] = (
        report["gates"]["promotion_ready"]
        and report["gates"]["vector_furniture_pass"]
    )
    report.update(
        {
            "dense_report": str(args.dense),
            "query_report": str(args.query),
            "dense_eval": str(args.dense_eval),
            "query_eval": str(args.query_eval),
            "test_opened": False,
            "scope": "matched_64_train_32_val_eight_epoch",
            "vector_metrics_required_before_final_decision": False,
            "parameter_confounder": (
                "query arm adds parameters and a furniture-query loss; "
                "a positive result needs matched-capacity follow-up"
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
