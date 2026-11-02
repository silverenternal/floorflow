#!/usr/bin/env python3
"""Aggregate five-seed canonical opening-consumer promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.opening_consumer_comparison import (
    compare_opening_reports,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "cal384": "cal384_structure_v1",
    "holdout107": "holdout107_structure_v1",
    "val491": "val491_structure_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 20, 21, 22, 23])
    parser.add_argument("--report-root", type=Path, default=ROOT / "outputs/reports")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/opening_interval_queryblend_fiveseed_v1.json",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_result(
    seed: int,
    suffix: str,
    root: Path,
    iterations: int,
    comparison_seed: int,
) -> dict:
    baseline_path = root / f"dense_query_blend_seed{seed}_structure_val491_v1.json"
    candidate_path = (
        root / f"opening_interval_queryblend_seed{seed}_{suffix}.json"
    )
    comparison = compare_opening_reports(
        _load(baseline_path),
        _load(candidate_path),
        iterations=iterations,
        seed=comparison_seed,
    )
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        **comparison,
    }


def _aggregate(seed_results: dict[str, dict]) -> dict:
    opening = [row["paired"]["fpeval_opening_f1"] for row in seed_results.values()]
    deltas = np.asarray([row["mean_delta"] for row in opening])
    return {
        "seeds": sorted(int(seed) for seed in seed_results),
        "opening_delta_mean": float(deltas.mean()),
        "opening_delta_std": float(deltas.std(ddof=1)),
        "opening_delta_min": float(deltas.min()),
        "opening_delta_max": float(deltas.max()),
        "opening_improved_seeds": int(
            sum(row["mean_delta"] > 0.0 for row in opening)
        ),
        "opening_ci_excludes_zero_seeds": int(
            sum(row["ci_low"] > 0.0 for row in opening)
        ),
        "non_regression_seeds": int(
            sum(row["gates"]["passed"] for row in seed_results.values())
        ),
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    splits = {
        name: {
            str(seed): _seed_result(
                seed,
                suffix,
                args.report_root,
                args.iterations,
                args.seed,
            )
            for seed in args.seeds
        }
        for name, suffix in SPLITS.items()
    }
    aggregates = {
        name: _aggregate(seed_rows)
        for name, seed_rows in splits.items()
    }
    gates = {
        name: {
            "opening_improved_at_least_four": row["opening_improved_seeds"] >= 4,
            "opening_ci_excludes_zero_at_least_four": (
                row["opening_ci_excludes_zero_seeds"] >= 4
            ),
            "non_regression_all_seeds": (
                row["non_regression_seeds"] == len(args.seeds)
            ),
        }
        for name, row in aggregates.items()
    }
    for value in gates.values():
        value["passed"] = all(value.values())
    report = {
        "schema_version": "opening-interval-fiveseed-v1",
        "protocol": "predicted_control",
        "target_used_at_inference": False,
        "test_opened": False,
        "expected_seeds": args.seeds,
        "aggregates": aggregates,
        "gates": gates,
        "decision": "promote" if all(row["passed"] for row in gates.values()) else "hold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
