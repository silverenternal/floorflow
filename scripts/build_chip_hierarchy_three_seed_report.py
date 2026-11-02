#!/usr/bin/env python3
"""Build the three-seed typed-hierarchy promotion report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

SEEDS = (19, 20, 21)
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(path: Path) -> list[dict]:
    return read(path)["nfe4_refine400"]


def paths(seed: int, use_ema: bool) -> tuple[Path, Path]:
    if use_ema:
        return (
            ROOT
            / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
            / f"candidate_seed{seed}_ema_primary_cases.json",
            ROOT
            / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
            / f"baseline_seed{seed}_ema_primary_cases.json",
        )
    if seed == 19:
        return (
            ROOT
            / "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
            "reproducible_v1/candidate_projector8_primary_cases.json",
            ROOT
            / "outputs/chip_hierarchy/v4_20k/"
            "reproducible_v1/baseline_primary_cases.json",
        )
    return (
        ROOT
        / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
        / f"candidate_seed{seed}_primary_cases.json",
        ROOT
        / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
        / f"baseline_seed{seed}_primary_cases.json",
    )


def means(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_typed_hierarchy_three_seed_v2.json",
    )
    args = parser.parse_args()
    results = {}
    rows_by_seed = {}
    for seed in SEEDS:
        candidate_path, baseline_path = paths(seed, args.use_ema)
        candidate = load_cases(candidate_path)
        baseline = load_cases(baseline_path)
        rows_by_seed[seed] = (candidate, baseline)
        results[str(seed)] = {
            "candidate": means(candidate),
            "baseline": means(baseline),
            "absolute_delta": {
                metric: means(candidate)[metric] - means(baseline)[metric]
                for metric in METRICS
            },
            "relative_delta": {
                metric: means(candidate)[metric] / means(baseline)[metric] - 1.0
                for metric in METRICS
            },
            "paired": {
                metric: paired_delta(
                    candidate,
                    baseline,
                    metric,
                    iterations=4000,
                    seed=20261500 + seed * 20 + index,
                    lower_is_better=True,
                )
                for index, metric in enumerate(METRICS)
            },
            "candidate_legalized_feasible_fraction": float(
                np.mean([bool(row["legalized_feasible"]) for row in candidate])
            ),
            "baseline_legalized_feasible_fraction": float(
                np.mean([bool(row["legalized_feasible"]) for row in baseline])
            ),
        }
    seed_mean = {
        "candidate": {
            metric: float(
                np.mean(
                    [
                        results[str(seed)]["candidate"][metric]
                        for seed in SEEDS
                    ]
                )
            )
            for metric in METRICS
        },
        "baseline": {
            metric: float(
                np.mean(
                    [
                        results[str(seed)]["baseline"][metric]
                        for seed in SEEDS
                    ]
                )
            )
            for metric in METRICS
        },
    }
    seed_mean["relative_delta"] = {
        metric: seed_mean["candidate"][metric]
        / seed_mean["baseline"][metric]
        - 1.0
        for metric in METRICS
    }
    pooled_candidate = [
        row
        for seed in SEEDS
        for row in rows_by_seed[seed][0]
    ]
    pooled_baseline = [
        row
        for seed in SEEDS
        for row in rows_by_seed[seed][1]
    ]
    pooled = {
        metric: paired_delta(
            pooled_candidate,
            pooled_baseline,
            metric,
            iterations=4000,
            seed=20261600 + index,
            lower_is_better=True,
        )
        for index, metric in enumerate(METRICS)
    }
    checks = {
        "all_seeds_legalized_feasible": all(
            results[str(seed)]["candidate_legalized_feasible_fraction"] == 1.0
            for seed in SEEDS
        ),
        "seed_mean_cost_improves": (
            seed_mean["relative_delta"]["legalized_hard_cost"] < 0.0
        ),
        "seed_mean_v_rel_does_not_regress": (
            seed_mean["relative_delta"]["legalized_v_rel"] <= 0.0
        ),
        "seed_mean_b2b_improves": (
            seed_mean["relative_delta"]["legalized_b2b_hpwl_gap"] < 0.0
        ),
        "seed_mean_area_improves": (
            seed_mean["relative_delta"]["legalized_area_gap"] < 0.0
        ),
        "at_least_two_seeds_cost_improve": sum(
            results[str(seed)]["relative_delta"]["legalized_hard_cost"] < 0.0
            for seed in SEEDS
        ) >= 2,
        "at_least_two_seeds_b2b_improve": sum(
            results[str(seed)]["relative_delta"]["legalized_b2b_hpwl_gap"]
            < 0.0
            for seed in SEEDS
        ) >= 2,
    }
    report = {
        "status": (
            "pass" if all(checks.values()) else "mixed_or_rejected"
        ),
        "scope": "three-seed 20k typed hierarchy versus graph-token control",
        "primary_endpoint": "NFE4 + 400 refinement + deterministic scale_pack",
        "candidate_projector_iterations": 8,
        "seeds": results,
        "seed_mean": seed_mean,
        "pooled_paired": pooled,
        "checks": checks,
        "best_of_n_used": False,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "seed_mean": seed_mean["relative_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
