#!/usr/bin/env python3
"""Build the three-seed M48 portfolio versus exact-anchor report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m48_portfolio_three_seed_v1.json",
    )
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed(seed: int) -> tuple[list[dict], list[dict], dict, dict]:
    suffix = "v1" if seed == 19 else f"seed{seed}"
    candidate = ROOT / f"outputs/reports/chip_m48_portfolio_official100_{suffix}.json"
    anchor = ROOT / (
        f"outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_{suffix}.json"
    )
    return (
        read(candidate)["rows"],
        read(anchor)["rows"],
        read(candidate),
        read(anchor),
    )


COMPONENTS = {
    "cost": ("cost", "cost"),
    "v_rel": ("v_rel", "v_rel"),
    "b2b_hpwl_gap": ("b2b_hpwl_gap", "legalized_b2b_hpwl_gap"),
    "p2b_hpwl_gap": ("p2b_hpwl_gap", "legalized_p2b_hpwl_gap"),
    "area_gap": ("area_gap", "legalized_area_gap"),
    "boundary": ("boundary", "legalized_boundary"),
    "grouping": ("grouping", "legalized_grouping"),
    "rho_mae": ("rho_mae", "legalized_rho_mae"),
}


def paired(
    candidate: list[dict],
    anchor: list[dict],
    metric: str,
) -> dict:
    candidate_key, anchor_key = COMPONENTS[metric]
    current = [
        {
            "index": int(row["index"]),
            "value": float(row["selected"]["stages"]["legalized"][candidate_key]),
        }
        for row in candidate
    ]
    reference = [
        {"index": int(row["index"]), "value": float(row[anchor_key])}
        for row in anchor
    ]
    return paired_delta(
        current,
        reference,
        "value",
        iterations=10000,
        lower_is_better=True,
    )


def main() -> None:
    args = parse_args()
    seeds = {}
    pooled_candidate = []
    pooled_anchor = []
    for seed in (19, 20, 21):
        candidate, anchor, candidate_report, anchor_report = load_seed(seed)
        if [row["index"] for row in candidate] != [row["index"] for row in anchor]:
            raise ValueError(f"seed {seed} candidate/anchor indices differ")
        selection_counts = Counter(row["selection"] for row in candidate)
        seeds[str(seed)] = {
            "candidate_cost": float(
                np.mean(
                    [row["selected"]["stages"]["legalized"]["cost"] for row in candidate]
                )
            ),
            "candidate_v_rel": float(
                np.mean(
                    [
                        row["selected"]["stages"]["legalized"]["v_rel"]
                        for row in candidate
                    ]
                )
            ),
            "anchor_cost": float(np.mean([row["cost"] for row in anchor])),
            "anchor_v_rel": float(np.mean([row["v_rel"] for row in anchor])),
            "selection_counts": dict(selection_counts),
            "paired_cost": paired(candidate, anchor, "cost"),
            "paired_v_rel": paired(candidate, anchor, "v_rel"),
            "feasible_fraction": float(
                np.mean(
                    [
                        row["selected"]["stages"]["legalized"]["feasible"]
                        for row in candidate
                    ]
                )
            ),
            "solver_escalations": candidate_report["summary"]["solver_escalations"],
            "candidate_wall_clock_seconds": candidate_report["summary"][
                "wall_clock_seconds"
            ],
            "anchor_mean_seconds": anchor_report["mean_seconds"],
        }
        for candidate_row, anchor_row in zip(candidate, anchor, strict=True):
            base = {
                "index": int(candidate_row["index"]),
                "seed": seed,
            }
            pooled_candidate.append(
                base
                | {
                    "cost": float(
                        candidate_row["selected"]["stages"]["legalized"]["cost"]
                    ),
                    "v_rel": float(
                        candidate_row["selected"]["stages"]["legalized"]["v_rel"]
                    ),
                }
            )
            pooled_anchor.append(
                base
                | {
                    "cost": float(anchor_row["cost"]),
                    "v_rel": float(anchor_row["v_rel"]),
                }
            )
    pooled = {}
    for metric in ("cost", "v_rel"):
        current = [
            {"index": row["index"], "seed": row["seed"], "value": row[metric]}
            for row in pooled_candidate
        ]
        reference = [
            {"index": row["index"], "seed": row["seed"], "value": row[metric]}
            for row in pooled_anchor
        ]
        pooled[metric] = paired_delta(
            current,
            reference,
            "value",
            iterations=10000,
            lower_is_better=True,
        )
    pass_gate = (
        all(row["feasible_fraction"] == 1.0 for row in seeds.values())
        and all(not row["solver_escalations"] for row in seeds.values())
        and pooled["cost"]["relative_delta"] < 0.0
        and pooled["cost"]["ci_high"] < 0.0
        and pooled["v_rel"]["relative_delta"] < 0.0
        and pooled["v_rel"]["ci_high"] < 0.0
    )
    report = {
        "version": "chip-m48-portfolio-three-seed-v1",
        "status": (
            "seed_stable_absolute_budget_pass"
            if pass_gate
            else "seed_stable_absolute_budget_rejected"
        ),
        "protocol": {
            "benchmark": "Official FloorSet 100",
            "seeds": [19, 20, 21],
            "nfe_per_trajectory": 1,
            "trajectories": 2,
            "refiner_steps_per_trajectory": 1600,
            "legalizer_calls": 3,
            "target_free_selection": True,
        },
        "budget_class": "absolute_2x_generation_not_same_inference",
        "seeds": seeds,
        "pooled": pooled,
        "claim_boundary": (
            "Seed-stable mean improvement against exact Trinity NFE1/ref1600 "
            "under a declared 2x generation budget. Same-inference, external "
            "routed-PPA and universal subgroup claims remain unsupported."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **pooled}, indent=2))


if __name__ == "__main__":
    main()
