#!/usr/bin/env python3
"""Decompose sparse consensus against every fixed candidate view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
    "legalizer_rung",
    "legalizer_solves",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20500)
    return parser.parse_args()


def _rows(rows: list[dict], layout: str) -> list[dict]:
    return [
        {
            "index": int(row["case_index"]),
            **{metric: float(row["rows"][layout][metric]) for metric in METRICS},
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    report = json.loads(args.validation.read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation report requires non-empty rows")
    candidates = [
        key
        for key in rows[0]["rows"]
        if key.startswith("candidate_") and key != "consensus"
    ]
    consensus = _rows(rows, "consensus")
    comparisons = {}
    for offset, candidate in enumerate(candidates):
        baseline = _rows(rows, candidate)
        comparisons[candidate] = {
            metric: paired_delta(
                consensus,
                baseline,
                metric,
                iterations=args.bootstrap,
                seed=args.seed + offset * len(METRICS) + metric_index,
                lower_is_better=True,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    result = {
        "version": "chip-candidate-consensus-validation-analysis-v1",
        "status": "complete",
        "source": str(args.validation),
        "cases": len(rows),
        "candidates": candidates,
        "candidate_means": {
            candidate: {
                metric: float(
                    np.mean([float(row["rows"][candidate][metric]) for row in rows])
                )
                for metric in METRICS
            }
            for candidate in candidates
        },
        "consensus_means": {
            metric: float(
                np.mean([float(row["rows"]["consensus"][metric]) for row in rows])
            )
            for metric in METRICS
        },
        "paired_vs_fixed_candidates": comparisons,
        "oracle_best_single_fraction": float(
            np.mean(
                [
                    row["rows"][row["best_single"]]["legalized_hard_cost"]
                    == row["rows"]["consensus"]["legalized_hard_cost"]
                    for row in rows
                ]
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": result["cases"],
                "consensus_means": result["consensus_means"],
                "oracle_best_single_fraction": result["oracle_best_single_fraction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
