#!/usr/bin/env python3
"""Build the three-seed flat-versus-global report through the same refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_global_vs_flat_refined_seeds_v1.json",
    )
    return parser.parse_args()


def read_rows(path: Path, key: str | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        payload = payload[key]
    return payload


def paired(
    control: list[dict],
    candidate: list[dict],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict:
    left = {int(row["index"]): row for row in control}
    right = {int(row["index"]): row for row in candidate}
    indices = sorted(set(left) & set(right))
    delta = np.asarray(
        [
            float(right[index][metric]) - float(left[index][metric])
            for index in indices
        ]
    )
    generator = np.random.default_rng(seed)
    draws = np.asarray(
        [
            generator.choice(delta, len(delta), replace=True).mean()
            for _ in range(iterations)
        ]
    )
    control_mean = float(
        np.mean([float(left[index][metric]) for index in indices])
    )
    candidate_mean = float(
        np.mean([float(right[index][metric]) for index in indices])
    )
    return {
        "cases": len(indices),
        "control_mean": control_mean,
        "candidate_mean": candidate_mean,
        "paired_delta": float(delta.mean()),
        "relative_delta": (candidate_mean - control_mean) / control_mean,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    refined_root = ROOT / "outputs/chip_refined_seeds"
    paths = {
        "flat19": refined_root / "flat19_nfe4_ref400_cases.json",
        "flat20": refined_root / "flat20_nfe4_ref400_cases.json",
        "flat21": refined_root / "flat21_nfe4_ref400_cases.json",
        "global20": refined_root / "global20_nfe4_ref400_cases.json",
        "global21": refined_root / "global21_nfe4_ref400_cases.json",
    }
    rows = {
        name: read_rows(path, "nfe4_refine400")
        for name, path in paths.items()
    }
    rows["global19"] = read_rows(
        ROOT / "outputs/reports/chip_inference_refiner_comparison_v1_cases.json",
        "nfe4_refine400",
    )
    results = {}
    for seed in (19, 20, 21):
        results[str(seed)] = {
            metric: paired(
                rows[f"flat{seed}"],
                rows[f"global{seed}"],
                metric,
                iterations=args.bootstrap,
                seed=20261027 + seed,
            )
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
    for seed, result in results.items():
        result["both_metrics_improve"] = all(
            result[metric]["paired_delta"] < 0.0
            and result[metric]["ci95"][1] < 0.0
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        )
    means = {
        metric: float(
            np.mean(
                [
                    result[metric]["relative_delta"]
                    for result in results.values()
                ]
            )
        )
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "all_seed_reports_complete": all(
            result["legalized_hard_cost"]["cases"] == 100
            for result in results.values()
        ),
        "global_wins_all_seeds": all(
            result["both_metrics_improve"] for result in results.values()
        ),
        "mean_hard_cost_improves": means["legalized_hard_cost"] < 0.0,
        "mean_v_rel_improves": means["legalized_v_rel"] < 0.0,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "single-sample NFE4 + 400-step closed-form refiner; score_full_fast",
        "protocol": {
            "samples_per_case": 1,
            "nfe": 4,
            "refiner_steps": 400,
            "legalizer": "scale_pack",
            "scorer": "score_full_fast",
            "best_of_n_used": False,
            "target_derived_control": False,
        },
        "checks": checks,
        "seeds": results,
        "seed_mean_relative_delta": means,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
