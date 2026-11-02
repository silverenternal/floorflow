#!/usr/bin/env python3
"""Audit seed heterogeneity without pseudo-replicating repeated cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import clustered_paired_delta, paired_delta

ROOT = Path(__file__).resolve().parents[1]
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
SMALLEST_EFFECTS = {
    "legalized_hard_cost": -0.02,
    "legalized_v_rel": -0.03,
    "legalized_hpwl_gap": -0.02,
    "legalized_area_gap": -0.02,
    "legalized_b2b_hpwl_gap": -0.02,
    "legalized_p2b_hpwl_gap": -0.02,
    "legalized_grouping": -0.02,
    "legalized_boundary": -0.02,
    "legalized_rho_mae": -0.02,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_seed_aware_statistics_v1.json",
    )
    parser.add_argument("--iterations", type=int, default=4000)
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paths(mode: str, arm: str, seed: int) -> Path:
    root = ROOT / "outputs/chip_hierarchy/v10_three_seed/reproducible_v1"
    file_arm = "baseline" if arm == "control" else "candidate"
    if seed == 19 and arm == "candidate" and mode != "ema":
        return (
            ROOT
            / "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
            "reproducible_v1/candidate_projector8_primary_cases.json"
        )
    if seed == 19 and arm == "control" and mode != "ema":
        return (
            ROOT
            / "outputs/chip_hierarchy/v4_20k/"
            "reproducible_v1/baseline_primary_cases.json"
        )
    stem = (
        f"{file_arm}_seed{seed}_{mode}_primary_cases.json"
        if mode == "ema"
        else f"{file_arm}_seed{seed}_primary_cases.json"
    )
    return root / stem


def load_rows(mode: str, arm: str, seed: int) -> list[dict]:
    return read(paths(mode, arm, seed))["nfe4_refine400"]


def metric_array(rows: list[dict], metric: str) -> np.ndarray:
    return np.asarray([float(row[metric]) for row in rows], dtype=np.float64)


def cluster_bootstrap(
    candidate_by_seed: dict[int, list[dict]],
    control_by_seed: dict[int, list[dict]],
    metric: str,
    iterations: int,
    seed: int,
) -> dict:
    result = clustered_paired_delta(
        [
            row
            for index in SEEDS
            for row in candidate_by_seed[index]
        ],
        [
            row
            for index in SEEDS
            for row in control_by_seed[index]
        ],
        metric,
        cluster_key="index",
        iterations=iterations,
        seed=seed,
    )
    result["seeds"] = list(SEEDS)
    result["improved_cases"] = result.pop("improved")
    result["worsened_cases"] = result.pop("worsened")
    result["unchanged_cases"] = result.pop("unchanged")
    result["bootstrap_unit"] = "official-case cluster with all three fixed seeds"
    return result


def analyze_mode(mode: str, iterations: int) -> dict:
    candidate = {seed: load_rows(mode, "candidate", seed) for seed in SEEDS}
    control = {seed: load_rows(mode, "control", seed) for seed in SEEDS}
    indices = {
        seed: [int(row["index"]) for row in candidate[seed]]
        for seed in SEEDS
    }
    if not all(indices[seed] == indices[19] for seed in SEEDS):
        raise ValueError("external case indices differ across seeds")
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            metric: {
                "candidate_mean": float(metric_array(candidate[seed], metric).mean()),
                "control_mean": float(metric_array(control[seed], metric).mean()),
                "relative_delta": float(
                    metric_array(candidate[seed], metric).mean()
                    / metric_array(control[seed], metric).mean()
                    - 1.0
                ),
                "paired": paired_delta(
                    candidate[seed],
                    control[seed],
                    metric,
                    iterations=iterations,
                    seed=20261700 + seed * 20 + metric_index,
                    lower_is_better=True,
                ),
            }
            for metric_index, metric in enumerate(METRICS)
        }
    seed_mean_relative = {}
    seed_mean_paired = {}
    leave_one_seed_out = {}
    for metric_index, metric in enumerate(METRICS):
        seed_effects = np.asarray(
            [
                per_seed[str(seed)][metric]["relative_delta"]
                for seed in SEEDS
            ]
        )
        seed_mean_relative[metric] = {
            "mean": float(seed_effects.mean()),
            "min": float(seed_effects.min()),
            "max": float(seed_effects.max()),
            "positive_seeds": int((seed_effects > 0.0).sum()),
            "negative_seeds": int((seed_effects < 0.0).sum()),
        }
        seed_mean_paired[metric] = cluster_bootstrap(
            candidate,
            control,
            metric,
            iterations,
            20261800 + metric_index,
        )
        leave_one_seed_out[metric] = {
            str(left_out): float(
                np.mean(
                    [
                        per_seed[str(seed)][metric]["relative_delta"]
                        for seed in SEEDS
                        if seed != left_out
                    ]
                )
            )
            for left_out in SEEDS
        }
    return {
        "case_indices_identical_across_seeds": True,
        "per_seed": per_seed,
        "seed_mean_relative": seed_mean_relative,
        "case_cluster_seed_mean": seed_mean_paired,
        "leave_one_seed_out": leave_one_seed_out,
    }


def main() -> None:
    args = parse_args()
    ema = analyze_mode("ema", args.iterations)
    raw = analyze_mode("raw", args.iterations)
    checks = {
        "ema_case_indices_identical": ema["case_indices_identical_across_seeds"],
        "raw_case_indices_identical": raw["case_indices_identical_across_seeds"],
        "ema_cluster_intervals_computed": all(
            value["cases"] == 100
            for value in ema["case_cluster_seed_mean"].values()
        ),
        "raw_and_ema_retained": bool(raw and ema),
        "leave_one_seed_out_retained": all(
            len(value) == 3
            for value in ema["leave_one_seed_out"].values()
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "fixed-seed official-100 paired analysis",
        "smallest_effects_of_interest": SMALLEST_EFFECTS,
        "ema": ema,
        "raw": raw,
        "statistical_boundary": (
            "The primary interval resamples official-case clusters and keeps "
            "all three fixed seeds; it does not treat the 300 case-seed rows "
            "as independent observations."
        ),
        "claim_policy": (
            "Use seed-mean only when the point estimate clears the smallest "
            "effect and report every crossing interval and leave-one-seed-out "
            "range. With three seeds, no asymptotic seed-significance claim."
        ),
        "checks": checks,
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
                "ema_seed_mean": ema["seed_mean_relative"],
                "ema_case_cluster": ema["case_cluster_seed_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
