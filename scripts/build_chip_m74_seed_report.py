#!/usr/bin/env python3
"""Aggregate cross-checkpoint portfolio results across inference seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "cost",
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)
TRINITY = {
    19: "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_v1.json",
    20: "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_seed20.json",
    21: "outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_seed21.json",
}
CROSS = {
    19: "outputs/reports/chip_m73_cross_checkpoint_portfolio_official100_v1.json",
    20: "outputs/reports/chip_m74_cross_checkpoint_seed20_official100_v1.json",
    21: "outputs/reports/chip_m74_cross_checkpoint_seed21_official100_v1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m74_cross_checkpoint_seed_report_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def trinity_rows(path: str) -> list[dict]:
    return [
        {
            "index": int(row["index"]),
            "cost": float(row["legalized_hard_cost"]),
            "v_rel": float(row["legalized_v_rel"]),
            "area_gap": float(row["legalized_area_gap"]),
            "b2b_hpwl_gap": float(row["legalized_b2b_hpwl_gap"]),
            "p2b_hpwl_gap": float(row["legalized_p2b_hpwl_gap"]),
            "grouping": float(row["legalized_grouping"]),
            "boundary": float(row["legalized_boundary"]),
            "rho_mae": float(row["legalized_rho_mae"]),
            "feasible": bool(row["feasible"]),
        }
        for row in load(path)["rows"]
    ]


def seed_summary(
    candidate: list[dict],
    reference: list[dict],
    iterations: int,
    seed: int,
) -> dict:
    paired = {
        key: paired_delta(
            candidate,
            reference,
            key,
            iterations=iterations,
            seed=seed * 100 + offset,
            lower_is_better=True,
        )
        for offset, key in enumerate(METRICS)
    }
    return {
        "cases": len(candidate),
        "relative_means": {
            key: value["relative_delta"] for key, value in paired.items()
        },
        "paired": paired,
        "feasibility_one": all(bool(row["feasible"]) for row in candidate),
    }


def main() -> None:
    args = parse_args()
    seeds = {}
    pooled_candidate = []
    pooled_reference = []
    for seed in (19, 20, 21):
        cross = load(CROSS[seed])
        candidate = [
            {
                "index": int(row["index"]),
                **{key: float(row[key]) for key in METRICS},
                "feasible": bool(row["feasible"]),
            }
            for row in cross["selected_rows"]
        ]
        reference = trinity_rows(TRINITY[seed])
        seeds[str(seed)] = seed_summary(
            candidate,
            reference,
            args.bootstrap,
            seed,
        )
        seeds[str(seed)]["selection_counts"] = cross.get("selection_counts", {})
        for offset, (current_row, reference_row) in enumerate(
            zip(candidate, reference, strict=True)
        ):
            unique_index = seed * 1000 + offset
            pooled_candidate.append({**current_row, "index": unique_index})
            pooled_reference.append({**reference_row, "index": unique_index})
    pooled = seed_summary(
        pooled_candidate,
        pooled_reference,
        args.bootstrap,
        74,
    )
    relative_means = {
        str(seed): summary["relative_means"] for seed, summary in seeds.items()
    }
    joint_primary_seeds = sum(
        values["cost"] < 0.0 and values["v_rel"] < 0.0
        for values in relative_means.values()
    )
    checks = {
        "all_feasible": all(row["feasibility_one"] for row in seeds.values()),
        "cost_mean_better_in_at_least_two_seeds": sum(
            values["cost"] < 0.0 for values in relative_means.values()
        )
        >= 2,
        "v_rel_mean_better_in_at_least_two_seeds": sum(
            values["v_rel"] < 0.0 for values in relative_means.values()
        )
        >= 2,
        "joint_primary_mean_better_in_at_least_two_seeds": joint_primary_seeds >= 2,
        "pooled_cost_ci_excludes_zero": pooled["paired"]["cost"]["ci_high"] < 0.0,
        "pooled_v_rel_ci_excludes_zero": pooled["paired"]["v_rel"]["ci_high"] < 0.0,
        "pooled_grouping_ci_excludes_zero": (
            pooled["paired"]["grouping"]["ci_high"] < 0.0
        ),
        "pooled_area_ci_excludes_zero": (
            pooled["paired"]["area_gap"]["ci_high"] < 0.0
        ),
        "pooled_b2b_ci_excludes_zero": (
            pooled["paired"]["b2b_hpwl_gap"]["ci_high"] < 0.0
        ),
    }
    stable_primary = (
        checks["all_feasible"]
        and checks["joint_primary_mean_better_in_at_least_two_seeds"]
        and checks["pooled_cost_ci_excludes_zero"]
        and checks["pooled_v_rel_ci_excludes_zero"]
    )
    bounded_mechanism = (
        checks["all_feasible"]
        and checks["joint_primary_mean_better_in_at_least_two_seeds"]
        and checks["pooled_v_rel_ci_excludes_zero"]
    )
    claim_class = (
        "stable_joint_primary_candidate"
        if stable_primary
        else "bounded_mechanism_candidate"
        if bounded_mechanism
        else "unsupported"
    )
    status = (
        "pass_stable_primary"
        if stable_primary
        else "pass_bounded_mechanism"
        if bounded_mechanism
        else "fail"
    )
    report = {
        "version": "chip-m74-cross-checkpoint-seed-report-v1",
        "status": status,
        "claim_class": claim_class,
        "seeds": seeds,
        "pooled": pooled,
        "joint_primary_seed_count": joint_primary_seeds,
        "checks": checks,
        "budget": {
            "nfe_total": 10,
            "refiner_steps_total": 9600,
            "legalizer_calls": 9,
            "trajectories": 6,
        },
        "claim_boundary": (
            "Three inference seeds only. Training-seed stability and external "
            "routed PPA remain required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
