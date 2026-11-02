#!/usr/bin/env python3
"""Combine the full seed-19 holdout with the five-design seed replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed19",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all_seed19.json",
    )
    parser.add_argument(
        "--replication",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_seeds20_21.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all.json",
    )
    return parser.parse_args()


def read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    seed19 = read(args.seed19)
    replication = read(args.replication)
    comparisons = [
        {**row, "source": str(args.seed19.relative_to(ROOT))}
        for row in seed19["comparisons"]
        if row["status"] == "completed"
    ]
    comparisons.extend(
        {**row, "source": str(args.replication.relative_to(ROOT))}
        for row in replication["comparisons"]
        if row["status"] == "completed"
    )
    unique = {
        (row["case"], int(row["seed"])): row for row in comparisons
    }
    comparisons = sorted(
        unique.values(),
        key=lambda row: (row["case"], int(row["seed"])),
    )
    runs_by_key = {
        (row["case"], row["arm"], int(row["seed"])): row
        for report in (seed19, replication)
        for row in report.get("runs", [])
    }
    runs = sorted(
        runs_by_key.values(),
        key=lambda row: (row["case"], int(row["seed"]), row["arm"]),
    )
    wire = [
        float(row["detailedroute__route__wirelength"]["relative"])
        for row in comparisons
    ]
    congestion = [
        float(row["globalplace__gpl__routability__congestion"]["relative"])
        for row in comparisons
        if row["globalplace__gpl__routability__congestion"]["relative"] is not None
    ]
    timing = [
        float(row["finish__timing__setup__ws"]["candidate"])
        - float(row["finish__timing__setup__ws"]["control"])
        for row in comparisons
    ]
    power = [
        float(row["finish__power__total"]["relative"])
        for row in comparisons
    ]
    area = [
        float(row["finish__design__instance__area__stdcell"]["relative"])
        for row in comparisons
    ]
    cases = sorted({row["case"] for row in comparisons})
    seeds = sorted({int(row["seed"]) for row in comparisons})
    case_means = {
        case: float(
            np.mean(
                [
                    row["detailedroute__route__wirelength"]["relative"]
                    for row in comparisons
                    if row["case"] == case
                ]
            )
        )
        for case in cases
    }
    seed_means = {
        str(seed): float(
            np.mean(
                [
                    row["detailedroute__route__wirelength"]["relative"]
                    for row in comparisons
                    if int(row["seed"]) == seed
                ]
            )
        )
        for seed in seeds
    }
    counts = {
        "wins": sum(value < -1.0e-12 for value in wire),
        "ties": sum(abs(value) <= 1.0e-12 for value in wire),
        "losses": sum(value > 1.0e-12 for value in wire),
    }
    drc_zero = all(
        row["candidate_drc"] == 0 and row["control_drc"] == 0
        for row in comparisons
    )
    checks = {
        "at_least_15_pairs": len(comparisons) >= 15,
        "at_least_8_designs": len(cases) >= 8,
        "drc_zero_both_arms": drc_zero,
        "routed_wirelength_nonregression": bool(wire)
        and float(np.mean(wire)) <= 0.0,
        "congestion_mean_at_most_2pct": bool(congestion)
        and float(np.mean(congestion)) <= 0.02,
        "power_max_within_5pct": bool(power) and max(power) <= 0.05,
        "stdcell_area_max_within_5pct": bool(area) and max(area) <= 0.05,
        "timing_mean_nonregression_20ps": bool(timing)
        and float(np.mean(timing)) >= -0.02,
    }
    report = {
        "version": "2026-10-28-chip-external-combined-v44",
        "status": (
            "pass" if all(checks.values()) else "external_ppa_not_supported"
        ),
        "comparison": (
            "typed hierarchy candidate versus graph-token control, identical "
            "1600-step deterministic refiner and matched legalizer route"
        ),
        "cases": cases,
        "design_count": len(cases),
        "seeds": seeds,
        "completed_pairs": len(comparisons),
        "seed_coverage": {
            str(seed): sum(int(row["seed"]) == seed for row in comparisons)
            for seed in seeds
        },
        "summary": {
            "primary_relative_wirelength_mean": float(np.mean(wire)),
            "primary_relative_wirelength_median": float(np.median(wire)),
            "primary_relative_wirelength_range": [min(wire), max(wire)],
            "congestion_relative_mean": float(np.mean(congestion)),
            "timing_ws_delta_mean_ns": float(np.mean(timing)),
            "power_relative_max": max(power),
            "stdcell_area_relative_max": max(area),
        },
        "wirelength_win_tie_loss": counts,
        "case_relative_wirelength_means": case_means,
        "seed_relative_wirelength_means": seed_means,
        "checks": checks,
        "comparisons": comparisons,
        "runs": runs,
        "reporting_rules": [
            "All selected cases and completed pairs are retained.",
            "Seeds are repeated measurements within a design, not independent cases.",
            "Only eight eligible macro-capable ORFS designs were available after license and input audit.",
            "No missing pair is replaced by a favorable case.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_pairs": len(comparisons),
                "design_count": len(cases),
                "summary": report["summary"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
