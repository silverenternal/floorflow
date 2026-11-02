#!/usr/bin/env python3
"""Build paired statistics for a dense versus room-instance control report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "room_macro_iou",
    "instance_instance_f1_at_threshold",
    "instance_count_abs_error",
    "instance_matched_type_accuracy",
    "node_f1",
    "edge_f1",
    "ged",
    "plan_plausible",
    "wall_continuity_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0.0].sum())
    total = float(ranks.sum())
    return float((2.0 * positive - total) / total)


def _paired_rows(
    baseline: list[dict],
    candidate: list[dict],
) -> tuple[list[dict], list[dict]]:
    baseline_by_case = {str(row["case"]): row for row in baseline}
    candidate_by_case = {str(row["case"]): row for row in candidate}
    shared = sorted(set(baseline_by_case) & set(candidate_by_case))
    if not shared:
        raise ValueError("reports share no case ids")
    return (
        [candidate_by_case[name] for name in shared],
        [baseline_by_case[name] for name in shared],
    )


def main() -> None:
    args = _parse_args()
    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate, baseline = _paired_rows(
        baseline_report["cases"],
        candidate_report["cases"],
    )
    statistics = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                float(cur[metric]) - float(ref[metric])
                for cur, ref in zip(candidate, baseline, strict=True)
            ],
            dtype=np.float64,
        )
        try:
            p_value = float(wilcoxon(differences, zero_method="wilcox").pvalue)
        except ValueError:
            p_value = 1.0
        statistics[metric] = {
            **paired_delta(
                candidate,
                baseline,
                metric,
                iterations=args.bootstrap,
                seed=args.seed,
            ),
            "rank_biserial": _rank_biserial(differences),
            "wilcoxon_p": p_value,
        }
    output = {
        "schema_version": "predicted-control-affinity-stats-v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_config": baseline_report.get("config", {}),
        "candidate_config": candidate_report.get("config", {}),
        "cases": len(candidate),
        "statistics": statistics,
        "interpretation_boundary": (
            "Paired validation statistics only; no test split was opened."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["statistics"], indent=2))


if __name__ == "__main__":
    main()
