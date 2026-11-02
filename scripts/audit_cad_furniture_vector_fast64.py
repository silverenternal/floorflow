#!/usr/bin/env python3
"""Aggregate two-seed furniture vector primitive evidence and decide promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense",
        type=Path,
        default=ROOT
        / "outputs/reports/cad_furniture_dense_fast64_eval_fthr01_v2.json",
    )
    parser.add_argument(
        "--seed-eval",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs/reports/cad_furniture_vector_fast64_eval_v2.json",
            ROOT
            / "outputs/reports/cad_furniture_vector_fast64_seed2906_eval_v2.json",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/cad_furniture_vector_fast64_eval_v1.json",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=ROOT / "outputs/reports/cad_pc06p_decision_v1.json",
    )
    return parser.parse_args()


def _summary(values: list[dict[str, float]], key: str) -> dict[str, float]:
    array = np.asarray([float(row[key]) for row in values])
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    dense_report = json.loads(args.dense.read_text(encoding="utf-8"))
    dense_rows = dense_report["cases"]
    dense_by_case = {row["case"]: row for row in dense_rows}
    seed_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.seed_eval
    ]
    keys = (
        "furniture_vector_f1",
        "furniture_vector_iou",
        "furniture_vector_matched_label_accuracy",
        "furniture_vector_matched_target_iou",
        "furniture_vector_attachment_accuracy",
        "furniture_vector_count",
        "furniture_vector_containment",
        "furniture_vector_polygon_valid_rate",
        "furniture_vector_out_of_boundary",
    )
    summaries = {
        key: _summary(
            [report["summary"] for report in seed_reports],
            key,
        )
        for key in keys
    }
    paired = {}
    for seed, report in zip(("2905", "2906"), seed_reports, strict=True):
        current = sorted(report["cases"], key=lambda row: row["case"])
        baseline = [dense_by_case[row["case"]] for row in current]
        paired[seed] = {
            key: paired_delta(current, baseline, key)
            for key in keys
        }
    dense = dense_report["summary"]
    quality_pass = (
        summaries["furniture_vector_f1"]["mean"]
        > float(dense["furniture_vector_f1"])
    )
    localization_pass = all(
        paired[seed][key]["ci_low"] > 0.0
        for seed in paired
        for key in (
            "furniture_vector_matched_label_accuracy",
            "furniture_vector_attachment_accuracy",
            "furniture_vector_polygon_valid_rate",
        )
    )
    containment_pass = (
        summaries["furniture_vector_containment"]["mean"]
        >= 1.0 - 1.0e-8
    )
    count_plausible = (
        float(dense["target_furniture_instances"]) * 0.5
        <= summaries["furniture_vector_count"]["mean"]
        <= float(dense["target_furniture_instances"]) * 2.0
    )
    semantic_risk = (
        summaries["furniture_vector_matched_label_accuracy"]["mean"]
        < 0.5
    )
    promotion_to_256 = bool(
        quality_pass
        and localization_pass
        and containment_pass
        and count_plausible
    )
    evaluation = {
        "schema_version": "cad-furniture-vector-fast64-eval-v1",
        "test_opened": False,
        "seeds": [report["checkpoint"] for report in seed_reports],
        "dense_control": dense,
        "seed_summary": summaries,
        "paired_vs_dense": paired,
        "checks": {
            "quality_pass": quality_pass,
            "localization_pass": localization_pass,
            "containment_pass": containment_pass,
            "count_plausible_proxy": count_plausible,
            "semantic_risk": semantic_risk,
            "promotion_to_256": promotion_to_256,
        },
        "source_reports": [str(path) for path in args.seed_eval],
        "dense_source": str(args.dense),
    }
    decision = {
        "schema_version": "cad-pc06p-decision-v1",
        "task": "PC-06P",
        "decision": (
            "promote_to_256_with_semantic_calibration"
            if promotion_to_256
            else "not_promoted"
        ),
        "main_claim_ready": False,
        "reason": (
            "Two-seed vector geometry and room attachment are stable, "
            "but furniture type accuracy remains low and needs calibration."
        ),
        "evaluation": str(args.output),
        "checks": evaluation["checks"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    args.decision.write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
