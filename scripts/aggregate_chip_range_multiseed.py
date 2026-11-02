#!/usr/bin/env python3
"""Aggregate seed19-21 high-block range comparisons with case clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.paired_bootstrap import clustered_paired_delta

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("legalized_hard_cost", "legalized_v_rel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_range_multiseed_aggregate_v1.json",
    )
    parser.add_argument(
        "--profile",
        choices=("ref400", "ref1600"),
        default="ref400",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_rows(path: Path, key: str) -> list[dict]:
    rows = load(path)
    if key not in rows:
        raise ValueError(f"{path} lacks {key}")
    return rows[key]


def seed_paths(
    seed: int,
    profile: str = "ref400",
) -> tuple[Path, Path, Path]:
    if profile == "ref1600":
        if seed == 19:
            return (
                ROOT / "outputs/reports/chip_range_comparison_seed19_refine1600.json",
                ROOT / "outputs/chip_hierarchy/sota_subset_control_v2/eval_20000_refine1600_cases.json",
                ROOT / "outputs/chip_hierarchy/sota_range_control_v1/eval_20000_refine1600_cases.json",
            )
        root = ROOT / "outputs/chip_hierarchy/range_multiseed_v1" / f"seed{seed}"
        return (
            ROOT / "outputs/reports" / f"chip_range_comparison_seed{seed}_refine1600.json",
            root / "control" / "eval_20000_refine1600_cases.json",
            root / "range" / "eval_20000_refine1600_cases.json",
        )
    if seed == 19:
        comparison = ROOT / "outputs/reports/chip_range_control_comparison_v1.json"
        control = (
            ROOT
            / "outputs/chip_hierarchy/sota_subset_control_v2/eval_20000_cuda1_cases.json"
        )
        candidate = (
            ROOT
            / "outputs/chip_hierarchy/sota_range_control_v1/eval_20000_cases.json"
        )
        return comparison, control, candidate
    root = ROOT / "outputs/chip_hierarchy/range_multiseed_v1" / f"seed{seed}"
    return (
        ROOT / "outputs/reports" / f"chip_range_comparison_seed{seed}.json",
        root / "control" / "eval_20000_cases.json",
        root / "range" / "eval_20000_cases.json",
    )


def main() -> None:
    args = parse_args()
    key = "nfe4_refine1600" if args.profile == "ref1600" else "nfe4_refine400"
    rows = {}
    control_pool = []
    range_pool = []
    for seed in (19, 20, 21):
        comparison_path, control_path, candidate_path = seed_paths(
            seed,
            args.profile,
        )
        missing = [
            str(path)
            for path in (comparison_path, control_path, candidate_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"seed {seed} artifacts missing: {missing}")
        comparison = load(comparison_path)
        control_rows = case_rows(control_path, key)
        candidate_rows = case_rows(candidate_path, key)
        control_pool.extend(control_rows)
        range_pool.extend(candidate_rows)
        rows[seed] = {
            "report": str(comparison_path.relative_to(ROOT)),
            "cost_relative": comparison["paired"]["legalized_hard_cost"][
                "relative_delta"
            ],
            "v_rel_relative": comparison["paired"]["legalized_v_rel"][
                "relative_delta"
            ],
            "cost_ci": [
                comparison["paired"]["legalized_hard_cost"]["ci_low"],
                comparison["paired"]["legalized_hard_cost"]["ci_high"],
            ],
            "v_rel_ci": [
                comparison["paired"]["legalized_v_rel"]["ci_low"],
                comparison["paired"]["legalized_v_rel"]["ci_high"],
            ],
        }
    pooled = {
        metric: clustered_paired_delta(
            range_pool,
            control_pool,
            metric,
            cluster_key="index",
            iterations=10000,
            seed=20261031 + offset,
        )
        for offset, metric in enumerate(METRICS)
    }
    all_seed_means_improve = all(
        row["cost_relative"] < 0.0 and row["v_rel_relative"] < 0.0
        for row in rows.values()
    )
    pooled_pass = all(pooled[metric]["ci_high"] < 0.0 for metric in METRICS)
    report = {
        "version": f"chip-range-multiseed-aggregate-{args.profile}-v1",
        "status": (
            "pass_seed_stability"
            if all_seed_means_improve and pooled_pass
            else "reject_seed_stability"
        ),
        "seeds": rows,
        "seed_mean_relative": {
            "cost": sum(row["cost_relative"] for row in rows.values()) / len(rows),
            "v_rel": (
                sum(row["v_rel_relative"] for row in rows.values()) / len(rows)
            ),
        },
        "seed_mean_improvements": {
            "cost": sum(row["cost_relative"] < 0.0 for row in rows.values()),
            "v_rel": sum(row["v_rel_relative"] < 0.0 for row in rows.values()),
        },
        "pooled_case_clustered": pooled,
        "checks": {
            "all_seed_means_improve_both": all_seed_means_improve,
            "pooled_cost_ci_excludes_zero": pooled["legalized_hard_cost"][
                "ci_high"
            ]
            < 0.0,
            "pooled_v_rel_ci_excludes_zero": pooled["legalized_v_rel"]["ci_high"]
            < 0.0,
        },
        "claim_boundary": (
            "Frozen 20k-subset seed replication only. Full-data and external "
            "routed-PPA gates remain required."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
