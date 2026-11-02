#!/usr/bin/env python3
"""Compare structural checkpoints through the existing M48 portfolio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta
from scripts.compare_chip_official_candidate import (
    load_candidate_rows,
    summarize_candidate,
)

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)
REFERENCE = "outputs/reports/chip_m48_portfolio_dev32_postfix_v1.json"
CANDIDATES = {
    "M61B": "outputs/reports/chip_m72_m61b_portfolio_dev32_v1.json",
    "M65": "outputs/reports/chip_m72_m65_portfolio_dev32_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m72_structural_portfolio_decision_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def summarize(report: dict) -> tuple[dict, list[dict]]:
    rows = load_candidate_rows(report, "portfolio")
    return summarize_candidate(report, "portfolio", rows), rows


def compare(
    candidate: list[dict],
    reference: list[dict],
    iterations: int,
) -> dict:
    return {
        metric: paired_delta(
            candidate,
            reference,
            metric,
            iterations=iterations,
            seed=2200 + offset,
            lower_is_better=True,
        )
        for offset, metric in enumerate(METRICS)
    }


def relative(candidate: dict, reference: dict) -> dict:
    return {
        metric: {
            "reference": float(reference[metric]),
            "candidate": float(candidate[metric]),
            "relative": float(candidate[metric]) / float(reference[metric]) - 1.0,
        }
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    reference_report = load(REFERENCE)
    reference_metrics, reference_rows = summarize(reference_report)
    results = {}
    for name, path in CANDIDATES.items():
        report = load(path)
        metrics, rows = summarize(report)
        paired = compare(rows, reference_rows, args.bootstrap)
        rel = relative(metrics, reference_metrics)
        results[name] = {
            "report": path,
            "selected_metrics": metrics,
            "relative_to_postfix_m48": rel,
            "paired": paired,
            "feasible_fraction": metrics["legalized_feasible_fraction"],
            "joint_primary_improves": (
                rel["legalized_hard_cost"]["relative"] < 0.0
                and rel["legalized_v_rel"]["relative"] < 0.0
            ),
            "structural_target_improves": any(
                rel[metric]["relative"] < 0.0
                for metric in (
                    "legalized_area_gap",
                    "legalized_b2b_hpwl_gap",
                    "legalized_boundary",
                )
            ),
        }
    ranked = sorted(
        results,
        key=lambda name: (
            not results[name]["joint_primary_improves"],
            results[name]["selected_metrics"]["legalized_hard_cost"],
            results[name]["selected_metrics"]["legalized_v_rel"],
        ),
    )
    selected = ranked[0] if results[ranked[0]]["joint_primary_improves"] else None
    report = {
        "version": "chip-m72-structural-portfolio-screen-v1",
        "status": "pass" if selected is not None else "fail",
        "protocol": {
            "cases": 32,
            "nfe_per_trajectory": 4,
            "refiner_steps_per_trajectory": 400,
            "trajectories": 2,
            "legalizer_calls": 3,
        },
        "reference": {
            "report": REFERENCE,
            "metrics": reference_metrics,
        },
        "candidates": results,
        "selected": selected,
        "claim_boundary": (
            "Small-screen composition only. A pass must be retrained or confirmed "
            "on Official-100 before any SOTA claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
