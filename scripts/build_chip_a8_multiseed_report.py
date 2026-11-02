#!/usr/bin/env python3
"""Aggregate the 200k data-scaling three-seed matched comparison."""

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
TRINITY_ANCHOR = {
    "cost": 1.0880284477882438,
    "v_rel": 0.023982607718226377,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-template",
        default=(
            "outputs/reports/chip_a8_200k_scaling_seed{seed}_"
            "20k_refine1600_dev.json"
        ),
    )
    parser.add_argument(
        "--control-template",
        default=(
            "outputs/reports/chip_a8_50k_control_seed{seed}_"
            "20k_refine1600_dev.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_a8_multiseed_report.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cases(path: Path) -> list[dict]:
    return read(path.with_name(f"{path.stem}_cases.json"))["nfe4_refine1600"]


def means(rows: list[dict]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    results = {}
    candidate_by_seed = {}
    control_by_seed = {}
    for seed in SEEDS:
        candidate_path = ROOT / args.candidate_template.format(seed=seed)
        control_path = ROOT / args.control_template.format(seed=seed)
        candidate = cases(candidate_path)
        control = cases(control_path)
        candidate_by_seed[seed] = candidate
        control_by_seed[seed] = control
        candidate_mean = means(candidate)
        control_mean = means(control)
        results[str(seed)] = {
            "candidate": candidate_mean,
            "control": control_mean,
            "relative_delta": {
                metric: candidate_mean[metric] / control_mean[metric] - 1.0
                for metric in METRICS
            },
            "paired": {
                metric: paired_delta(
                    candidate,
                    control,
                    metric,
                    iterations=args.bootstrap,
                    seed=20281000 + seed * 20 + position,
                    lower_is_better=True,
                )
                for position, metric in enumerate(METRICS)
            },
            "feasible_fraction": float(
                np.mean([bool(row["legalized_feasible"]) for row in candidate])
            ),
        }
    seed_mean = {
        arm: {
            metric: float(
                np.mean(
                    [results[str(seed)][arm][metric] for seed in SEEDS]
                )
            )
            for metric in METRICS
        }
        for arm in ("candidate", "control")
    }
    seed_mean["relative_delta"] = {
        metric: seed_mean["candidate"][metric] / seed_mean["control"][metric] - 1.0
        for metric in METRICS
    }
    pooled_candidate = [
        row for seed in SEEDS for row in candidate_by_seed[seed]
    ]
    pooled_control = [
        row for seed in SEEDS for row in control_by_seed[seed]
    ]
    pooled = {
        metric: clustered_paired_delta(
            pooled_candidate,
            pooled_control,
            metric,
            cluster_key="index",
            iterations=args.bootstrap,
            seed=20282000 + position,
        )
        for position, metric in enumerate(METRICS)
    }
    seed_checks = {
        str(seed): {
            "cost_improves": (
                results[str(seed)]["relative_delta"]["legalized_hard_cost"]
                < 0.0
            ),
            "v_rel_improves": (
                results[str(seed)]["relative_delta"]["legalized_v_rel"]
                < 0.0
            ),
            "feasible_fraction_one": (
                results[str(seed)]["feasible_fraction"] == 1.0
            ),
        }
        for seed in SEEDS
    }
    checks = {
        "all_seeds_improve_both": all(
            values["cost_improves"]
            and values["v_rel_improves"]
            and values["feasible_fraction_one"]
            for values in seed_checks.values()
        ),
        "pooled_cost_ci_excludes_zero": (
            pooled["legalized_hard_cost"]["ci_high"] < 0.0
        ),
        "pooled_v_rel_ci_excludes_zero": (
            pooled["legalized_v_rel"]["ci_high"] < 0.0
        ),
        "seed_mean_cost_improves": (
            seed_mean["relative_delta"]["legalized_hard_cost"] < 0.0
        ),
        "seed_mean_v_rel_improves": (
            seed_mean["relative_delta"]["legalized_v_rel"] < 0.0
        ),
    }
    report = {
        "version": "chip-a8-multiseed-v1",
        "status": (
            "pass" if all(checks.values()) else "reject_seed_instability"
        ),
        "scope": (
            "200k-vs-50k data scaling, identical architecture/optimizer/steps, "
            "frozen 32-case legalized dev, inferred with seed19"
        ),
        "seeds": results,
        "seed_checks": seed_checks,
        "seed_mean": seed_mean,
        "pooled_cluster_bootstrap": pooled,
        "trinity_anchor": TRINITY_ANCHOR,
        "anchor_gap": {
            "cost_relative": (
                seed_mean["candidate"]["legalized_hard_cost"]
                / TRINITY_ANCHOR["cost"]
                - 1.0
            ),
            "v_rel_relative": (
                seed_mean["candidate"]["legalized_v_rel"]
                / TRINITY_ANCHOR["v_rel"]
                - 1.0
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "No SOTA claim. External routed-PPA and full multiseed anchor gates "
            "remain separate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
