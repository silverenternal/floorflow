#!/usr/bin/env python3
"""Aggregate matched fixed-outline feasibility across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_outline_seed_stability_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(reports: list[tuple[int, dict]]) -> dict:
    rows = []
    candidate_only = []
    control_only = []
    paired = []
    for seed, report in reports:
        for case in report["cases"]:
            candidate = bool(case["candidate"]["selected"]["feasible"])
            control = bool(case["control"]["selected"]["feasible"])
            key = f"seed{seed}:{case['id']}"
            rows.append(
                {
                    "seed": seed,
                    "id": case["id"],
                    "candidate_feasible": candidate,
                    "control_feasible": control,
                }
            )
            if candidate and not control:
                candidate_only.append(key)
            if control and not candidate:
                control_only.append(key)
            if candidate and control:
                paired.append(
                    {
                        "seed": seed,
                        "id": case["id"],
                        "hpwl_relative": float(case["hpwl_relative"]),
                    }
                )
    discordant = len(candidate_only) + len(control_only)
    p_value = (
        binomtest(
            min(len(candidate_only), len(control_only)),
            discordant,
            0.5,
        ).pvalue
        if discordant
        else 1.0
    )
    report = {
        "version": "chip-outline-seed-stability-v1",
        "status": "structural_diagnostic_only",
        "seeds": [seed for seed, _ in reports],
        "rows": rows,
        "candidate_feasible": sum(row["candidate_feasible"] for row in rows),
        "control_feasible": sum(row["control_feasible"] for row in rows),
        "cases": len(rows),
        "candidate_only": candidate_only,
        "control_only": control_only,
        "feasibility_mcnemar_exact_p": float(p_value),
        "paired_feasible": paired,
        "paired_hpwl_relative_mean": (
            float(np.mean([row["hpwl_relative"] for row in paired]))
            if paired
            else None
        ),
        "claim_boundary": (
            "Two structural cases across three training seeds; this is a "
            "stability diagnostic, not a performance or routed-PPA claim."
        ),
    }
    return report


def main() -> None:
    args = parse_args()
    if args.report is None:
        args.report = [
            ROOT / "outputs/reports/chip_v46_bookshelf_outline_fallback_smoke_v7.json",
            ROOT / "outputs/reports/chip_v46_bookshelf_outline_fallback_seed20_smoke_v1.json",
            ROOT / "outputs/reports/chip_v46_bookshelf_outline_fallback_seed21_smoke_v1.json",
        ]
    if args.seed is None:
        args.seed = [19, 20, 21]
    if len(args.report) != len(args.seed):
        raise ValueError("one seed is required per report")
    result = analyze(
        [
            (seed, load(path))
            for seed, path in zip(args.seed, args.report, strict=True)
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
