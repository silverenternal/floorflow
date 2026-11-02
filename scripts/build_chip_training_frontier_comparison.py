#!/usr/bin/env python3
"""Compare same-device chip training-frontier evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_grouping",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_area_gap",
    "legalized_rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--case-key", default="nfe4_refine400")
    parser.add_argument("--scope", default="same-device frozen dev32 comparison")
    return parser.parse_args()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load(
    path: Path,
    case_key: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = read(path)
    cases_path = path.with_name(f"{path.stem}_cases.json")
    cases = read(cases_path)
    return report, report["results"][case_key], cases[case_key]


def comparison_rows(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: paired_delta(
            candidate,
            baseline,
            metric,
            iterations=iterations,
            seed=seed + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(METRICS)
    }


def mean_component(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    baseline = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(baseline.mean()),
        "p50": float(np.quantile(baseline, 0.50)),
        "p90": float(np.quantile(baseline, 0.90)),
        "p95": float(np.quantile(baseline, 0.95)),
    }


def main() -> None:
    args = parse_args()
    args.baseline = args.baseline.resolve()
    args.candidate = args.candidate.resolve()
    args.output = args.output.resolve()
    baseline_report, baseline_summary, baseline_rows = load(
        args.baseline,
        args.case_key,
    )
    candidate_report, candidate_summary, candidate_rows = load(
        args.candidate,
        args.case_key,
    )
    if baseline_report.get("device") != candidate_report.get("device"):
        raise ValueError(
            "same-device comparison requires identical eval device: "
            f"{baseline_report.get('device')} != {candidate_report.get('device')}"
        )
    paired = comparison_rows(
        baseline_rows,
        candidate_rows,
        iterations=args.bootstrap,
        seed=args.seed,
    )
    checks = {
        metric: (
            paired[metric]["mean_delta"] < 0.0
            and paired[metric]["ci_high"] < 0.0
        )
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    report = {
        "version": "chip-training-frontier-comparison-v1",
        "status": (
            "candidate_passes_same_device_gate"
            if all(checks.values())
            else "candidate_rejected_by_same_device_gate"
        ),
        "scope": args.scope,
        "device": baseline_report.get("device"),
        "case_key": args.case_key,
        "baseline": {
            "report": str(args.baseline.relative_to(ROOT)),
            "summary": baseline_summary,
            "tails": {
                metric: mean_component(baseline_rows, metric)
                for metric in ("legalized_hard_cost", "legalized_v_rel")
            },
        },
        "candidate": {
            "report": str(args.candidate.relative_to(ROOT)),
            "summary": candidate_summary,
            "tails": {
                metric: mean_component(candidate_rows, metric)
                for metric in ("legalized_hard_cost", "legalized_v_rel")
            },
        },
        "paired": paired,
        "primary_checks": checks,
        "decision": (
            "Advance only if both cost and V_rel improve with paired intervals "
            "below zero. Mixed or regressing arms remain completed evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
