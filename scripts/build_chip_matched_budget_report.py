#!/usr/bin/env python3
"""Compare candidate and Trinity d192 at the same refiner budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT
        / "outputs/chip_hierarchy/v44/refiner_official/"
        "candidate_boundary2_lp2s_1600_seed19.json",
    )
    parser.add_argument(
        "--trinity",
        type=Path,
        default=ROOT
        / "outputs/reports/trinity_d192_budget1600_weight2_seed19_v44.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_matched_budget_1600_seed19_v44.json",
    )
    parser.add_argument("--version", default="chip-matched-budget-v1")
    parser.add_argument(
        "--scope",
        default="Seed 19, Official-100, NFE4 + boundary=2 + 1600-step linear refiner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = (
        args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    ).resolve()
    trinity_path = (
        args.trinity if args.trinity.is_absolute() else ROOT / args.trinity
    ).resolve()
    candidate = read(candidate_path)
    if "results" in candidate:
        candidate_rows = read(
            candidate_path.with_name(f"{candidate_path.stem}_cases.json")
        )["nfe4_refine1600"]
        candidate_feasible = float(
            candidate["results"]["nfe4_refine1600"]["legalized_feasible_fraction"]
        )
    else:
        candidate_rows = candidate["rows"]
        candidate_feasible = float(candidate["feasible_fraction"])
    trinity = read(trinity_path)
    trinity_rows = trinity["rows"]
    metrics = (
        "legalized_hard_cost",
        "legalized_v_rel",
        "legalized_grouping",
        "legalized_boundary",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_area_gap",
    )
    candidate_mean = {
        name: float(np.mean([float(row[name]) for row in candidate_rows]))
        for name in metrics
    }
    trinity_mean = {
        name: float(np.mean([float(row[name]) for row in trinity_rows]))
        for name in metrics
    }
    candidate_refiner = candidate["refiner"]
    trinity_refiner = trinity["protocol"]["refiner"]
    same_budget = (
        int(
            candidate["refiner_steps"][0]
            if "refiner_steps" in candidate
            else candidate["protocol"]["steps"]
        )
        == int(trinity_refiner["steps"])
        == 1600
        and float(candidate_refiner["weights"].get("boundary", 1.0)) == 2.0
        and float(trinity_refiner["weights"]["boundary"]) == 2.0
    )
    pairs = {
        name: paired_delta(
            candidate_rows,
            trinity_rows,
            name,
            iterations=10000,
            seed=20269000 + index,
            lower_is_better=True,
        )
        for index, name in enumerate(metrics)
    }
    checks = {
        "same_refiner_budget": same_budget,
        "candidate_beats_cost": (
            candidate_mean["legalized_hard_cost"]
            < trinity_mean["legalized_hard_cost"]
        ),
        "candidate_beats_v_rel": (
            candidate_mean["legalized_v_rel"] < trinity_mean["legalized_v_rel"]
        ),
    }
    report = {
        "version": args.version,
        "status": (
            "candidate_win_at_matched_budget"
            if all(checks.values())
            else "matched_budget_boundary_only"
        ),
        "scope": args.scope,
        "candidate": {
            "path": str(candidate_path.relative_to(ROOT)),
            "mean": candidate_mean,
            "legalized_feasible_fraction": candidate_feasible,
            "device": candidate.get("device"),
        },
        "trinity": {
            "path": str(trinity_path.relative_to(ROOT)),
            "mean": trinity_mean,
            "feasible_fraction": float(trinity["feasible_fraction"]),
            "device": trinity.get("protocol", {}).get("device"),
            "schedule": trinity_refiner["lr_schedule"],
        },
        "candidate_schedule": candidate_refiner["schedule"],
        "paired": pairs,
        "checks": checks,
        "interpretation": (
            "A win supports a matched-budget quality claim on this seed. A loss "
            "means the fixed 400-step Trinity anchor comparison must be separated "
            "from any equal-compute claim."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
