#!/usr/bin/env python3
"""Build the paired 5k screen report for typed hierarchical chip flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-arm",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/v3_screen_5k/"
            "typed_hierarchy_seed19_5k"
        ),
    )
    parser.add_argument(
        "--baseline-arm",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_trinity/screen_5k/"
            "trinity_token_seed19_5k"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_typed_hierarchy_screen_5k_v2.json"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    points = np.quantile(values, [0.5, 0.75, 0.9, 0.95])
    return {
        name: float(value)
        for name, value in zip(("p50", "p75", "p90", "p95"), points)
    }


def comparisons(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    bootstrap: int,
) -> dict:
    output = {}
    for index, metric in enumerate(METRICS):
        candidate = np.asarray(
            [float(row[metric]) for row in candidate_rows],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [float(row[metric]) for row in baseline_rows],
            dtype=np.float64,
        )
        output[metric] = {
            "candidate_mean": float(candidate.mean()),
            "baseline_mean": float(baseline.mean()),
            "relative_delta": float(candidate.mean() / baseline.mean() - 1.0),
            "paired": paired_delta(
                candidate_rows,
                baseline_rows,
                metric,
                iterations=bootstrap,
                seed=20261110 + index,
                lower_is_better=True,
            ),
            "candidate_quantiles": quantiles(candidate_rows, metric),
            "baseline_quantiles": quantiles(baseline_rows, metric),
        }
    return output


def endpoint_curve(path: Path, key_prefix: str) -> dict:
    results = read(path)["results"]
    return {
        key: {
            metric: value[metric]
            for metric in METRICS + (
                "legalized_hard_cost_exp",
                "legalized_hard_cost_blockcount",
                "legalized_feasible_fraction",
            )
        }
        for key, value in results.items()
        if key.startswith(key_prefix)
    }


def main() -> None:
    args = parse_args()
    candidate_nfe = endpoint_curve(args.candidate_arm / "nfe_curve.json", "nfe")
    candidate_refiner = endpoint_curve(
        args.candidate_arm / "refiner_curve.json",
        "nfe4_refine",
    )
    baseline_raw = read(
        args.baseline_arm / "official_test100_nfe8_legalized.json"
    )
    baseline_refiner = read(
        args.baseline_arm / "refiner_nfe4_ref400.json"
    )["results"]["nfe4_refine400"]
    candidate_primary_rows = read(
        args.candidate_arm / "refiner_curve_cases.json"
    )["nfe4_refine400"]
    baseline_primary_rows = read(
        args.baseline_arm / "refiner_nfe4_ref400_cases.json"
    )["nfe4_refine400"]
    primary = comparisons(
        candidate_primary_rows,
        baseline_primary_rows,
        args.bootstrap,
    )
    raw_baseline_relative = {}
    for nfe, values in candidate_nfe.items():
        raw_baseline_relative[nfe] = {
            metric: values[metric] / float(baseline_raw[metric]) - 1.0
            for metric in METRICS
        }
    checks = {
        "legalized_feasible_fraction_one": all(
            float(values["legalized_feasible_fraction"]) == 1.0
            for values in (
                candidate_nfe["nfe4_refine0"],
                candidate_refiner["nfe4_refine400"],
            )
        ),
        "primary_cost_improves_at_least_5pct": (
            primary["legalized_hard_cost"]["relative_delta"] <= -0.05
        ),
        "primary_v_rel_improves_at_least_5pct": (
            primary["legalized_v_rel"]["relative_delta"] <= -0.05
        ),
        "primary_hpwl_does_not_regress": (
            primary["legalized_hpwl_gap"]["relative_delta"] <= 0.0
        ),
        "primary_area_does_not_regress": (
            primary["legalized_area_gap"]["relative_delta"] <= 0.0
        ),
    }
    report = {
        "status": (
            "pass"
            if all(checks.values())
            else "partial_pass_targeted_rescue_required"
        ),
        "scope": "seed-19 5k matched parameter screen on FloorSet official-100",
        "candidate_arm": str(args.candidate_arm.relative_to(ROOT)),
        "baseline_arm": str(args.baseline_arm.relative_to(ROOT)),
        "candidate_parameters": 5_836_443,
        "baseline_parameters": 5_856_400,
        "parameter_relative_error": 0.0034077248821802186,
        "primary_endpoint": "NFE4 + 400-step closed-form refiner + scale_pack",
        "baseline_primary": {
            metric: baseline_refiner[metric]
            for metric in METRICS + ("legalized_feasible_fraction",)
        },
        "candidate_primary": candidate_refiner["nfe4_refine400"],
        "primary_comparisons": primary,
        "nfe_curve": candidate_nfe,
        "nfe_curve_relative_to_baseline_nfe8": raw_baseline_relative,
        "refiner_curve": candidate_refiner,
        "budget_diagnostic": {
            "candidate_refine50": candidate_refiner["nfe4_refine50"],
            "candidate_refine200": candidate_refiner["nfe4_refine200"],
            "baseline_refine400": {
                metric: baseline_refiner[metric]
                for metric in METRICS
            },
            "candidate_refine50_note": (
                "Cost is better than baseline refine400, but V_rel is worse; "
                "50 refinement steps are not claimed as equal quality."
            ),
        },
        "checks": checks,
        "decision": (
            "Do not scale the unchanged architecture to three seeds. "
            "Diagnose the cost/grouping/boundary shortfall, then run a "
            "conditional 20k rescue against the existing 20k control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "primary": {
                    metric: primary[metric]["relative_delta"]
                    for metric in METRICS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
