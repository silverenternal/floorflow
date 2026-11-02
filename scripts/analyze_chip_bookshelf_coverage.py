#!/usr/bin/env python3
"""Summarize feasibility coverage and feasible-pair HPWL for bookshelf eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261030)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(report: dict, *, iterations: int, seed: int) -> dict:
    cases = report["cases"]
    candidate_feasible = [
        case
        for case in cases
        if bool(case["candidate"]["selected"]["feasible"])
    ]
    control_feasible = [
        case
        for case in cases
        if bool(case["control"]["selected"]["feasible"])
    ]
    candidate_only = [
        case
        for case in cases
        if case["candidate"]["selected"]["feasible"]
        and not case["control"]["selected"]["feasible"]
    ]
    control_only = [
        case
        for case in cases
        if case["control"]["selected"]["feasible"]
        and not case["candidate"]["selected"]["feasible"]
    ]
    discordant = len(candidate_only) + len(control_only)
    mcnemar_p = (
        binomtest(
            min(len(candidate_only), len(control_only)),
            discordant,
            p=0.5,
        ).pvalue
        if discordant
        else 1.0
    )
    paired = [
        {
            "index": index,
            "candidate": float(case["candidate"]["selected"]["hpwl"]),
            "control": float(case["control"]["selected"]["hpwl"]),
        }
        for index, case in enumerate(cases)
        if case["candidate"]["selected"]["feasible"]
        and case["control"]["selected"]["feasible"]
    ]
    result = {
        "version": "chip-bookshelf-coverage-analysis-v1",
        "status": "structural_diagnostic_only",
        "cases": len(cases),
        "candidate_feasible": len(candidate_feasible),
        "control_feasible": len(control_feasible),
        "candidate_only": [case["id"] for case in candidate_only],
        "control_only": [case["id"] for case in control_only],
        "both_feasible": len(paired),
        "feasibility_mcnemar_exact_p": float(mcnemar_p),
        "feasible_pair_hpwl": (
            paired_delta(
                [
                    {"index": row["index"], "hpwl": row["candidate"]}
                    for row in paired
                ],
                [
                    {"index": row["index"], "hpwl": row["control"]}
                    for row in paired
                ],
                "hpwl",
                iterations=iterations,
                seed=seed,
                lower_is_better=True,
            )
            if paired
            else None
        ),
        "claim_boundary": (
            "Bookshelf feasibility coverage is structural external evidence only. "
            "HPWL is reported only on paired feasible cases and is not routed PPA."
        ),
    }
    result["variants"] = {}
    for variant in ("hard", "soft"):
        variant_cases = [
            case
            for case in cases
            if case["id"].split("/")[1].lower() == variant
        ]
        variant_paired = [
            {
                "index": index,
                "candidate": float(case["candidate"]["selected"]["hpwl"]),
                "control": float(case["control"]["selected"]["hpwl"]),
            }
            for index, case in enumerate(variant_cases)
            if case["candidate"]["selected"]["feasible"]
            and case["control"]["selected"]["feasible"]
        ]
        result["variants"][variant] = {
            "cases": len(variant_cases),
            "candidate_feasible": sum(
                bool(case["candidate"]["selected"]["feasible"])
                for case in variant_cases
            ),
            "control_feasible": sum(
                bool(case["control"]["selected"]["feasible"])
                for case in variant_cases
            ),
            "both_feasible": len(variant_paired),
            "feasible_pair_hpwl": (
                paired_delta(
                    [
                        {"index": row["index"], "hpwl": row["candidate"]}
                        for row in variant_paired
                    ],
                    [
                        {"index": row["index"], "hpwl": row["control"]}
                        for row in variant_paired
                    ],
                    "hpwl",
                    iterations=iterations,
                    seed=seed + 1,
                    lower_is_better=True,
                )
                if variant_paired
                else None
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    result = analyze(
        load(args.input),
        iterations=args.iterations,
        seed=args.seed,
    )
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
