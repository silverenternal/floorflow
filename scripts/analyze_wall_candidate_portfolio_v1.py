#!/usr/bin/env python3
"""Measure the oracle upper bound of choosing between two wall-plan arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-arm", default="raw_score_opening_safe_gate")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = json.loads(args.stage_report.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    stage_by_case = {row["case"]: row for row in stage["cases"]}
    candidate_rows = candidate["per_case"][args.candidate_arm]
    pairs = []
    for row in candidate_rows:
        stage_row = stage_by_case.get(row["case"])
        if stage_row is None:
            continue
        stage_f1 = float(stage_row["fpeval_wall_f1"])
        candidate_f1 = float(row["wall"])
        stage_wins = stage_f1 >= candidate_f1
        pairs.append(
            {
                "case": row["case"],
                "stage_wall_f1": stage_f1,
                "candidate_wall_f1": candidate_f1,
                "selected": "stage" if stage_wins else "candidate",
                "oracle_wall_f1": max(stage_f1, candidate_f1),
                "oracle_wall_precision": float(
                    stage_row["fpeval_wall_precision"]
                    if stage_wins
                    else row["wall_precision"]
                ),
                "oracle_wall_recall": float(
                    stage_row["fpeval_wall_recall"]
                    if stage_wins
                    else row["wall_recall"]
                ),
            }
        )
    if not pairs:
        raise ValueError("stage and candidate reports have no shared cases")
    values = {
        key: float(np.mean([row[key] for row in pairs]))
        for key in (
            "stage_wall_f1",
            "candidate_wall_f1",
            "oracle_wall_f1",
            "oracle_wall_precision",
            "oracle_wall_recall",
        )
    }
    report = {
        "schema_version": "wall-candidate-portfolio-upper-bound-v1",
        "scope": "per-case oracle selection; not deployable",
        "stage_report": str(args.stage_report),
        "candidate_report": str(args.candidate_report),
        "candidate_arm": args.candidate_arm,
        "cases": len(pairs),
        "candidate_wins": sum(row["selected"] == "candidate" for row in pairs),
        "stage_wins": sum(row["selected"] == "stage" for row in pairs),
        "metrics": values,
        "oracle_gain_over_stage": (
            values["oracle_wall_f1"] - values["stage_wall_f1"]
        ),
        "checks": {
            "oracle_wall_f1_at_least_0_58": values["oracle_wall_f1"] >= 0.58,
            "oracle_wall_precision_at_least_0_52": (
                values["oracle_wall_precision"] >= 0.52
            ),
            "oracle_wall_recall_at_least_0_65": (
                values["oracle_wall_recall"] >= 0.65
            ),
        },
        "decision": (
            "new_candidates_required"
            if values["oracle_wall_f1"] < 0.58
            else "selector_route_viable"
        ),
        "per_case": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
