#!/usr/bin/env python3
"""Compare fixed-outline feasibility with and without the U3 fallback."""

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
        "--fallback",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v46_bookshelf_outline_fallback_full22_v1.json",
    )
    parser.add_argument(
        "--no-fallback",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v46_bookshelf_no_fallback_full22_4draw_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_outline_fallback_effect_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(report: dict, arm: str, variant: str | None = None) -> dict:
    cases = [
        case
        for case in report["cases"]
        if variant is None or case["id"].split("/")[1].lower() == variant
    ]
    return {
        "cases": len(cases),
        "feasible": sum(case[arm]["selected"]["feasible"] for case in cases),
    }


def analyze(fallback: dict, no_fallback: dict) -> dict:
    fall_by_id = {case["id"]: case for case in fallback["cases"]}
    nofall_by_id = {case["id"]: case for case in no_fallback["cases"]}
    if set(fall_by_id) != set(nofall_by_id):
        raise ValueError("fallback and no-fallback reports cover different cases")
    output = {}
    for arm in ("candidate", "control"):
        transitions = []
        hpwl = []
        for case_id in sorted(fall_by_id):
            before = bool(nofall_by_id[case_id][arm]["selected"]["feasible"])
            after = bool(fall_by_id[case_id][arm]["selected"]["feasible"])
            transitions.append((before, after))
            if before and after:
                hpwl.append(
                    float(fall_by_id[case_id][arm]["selected"]["hpwl"])
                    / float(nofall_by_id[case_id][arm]["selected"]["hpwl"])
                    - 1.0
                )
        gain = sum(not before and after for before, after in transitions)
        loss = sum(before and not after for before, after in transitions)
        discordant = gain + loss
        gained_ids = [
            case_id
            for case_id in sorted(fall_by_id)
            if not nofall_by_id[case_id][arm]["selected"]["feasible"]
            and fall_by_id[case_id][arm]["selected"]["feasible"]
        ]
        lost_ids = [
            case_id
            for case_id in sorted(fall_by_id)
            if nofall_by_id[case_id][arm]["selected"]["feasible"]
            and not fall_by_id[case_id][arm]["selected"]["feasible"]
        ]
        output[arm] = {
            "with_fallback": summarize(fallback, arm),
            "without_fallback": summarize(no_fallback, arm),
            "gained_cases": gain,
            "lost_cases": loss,
            "gained_case_ids": gained_ids,
            "lost_case_ids": lost_ids,
            "mcnemar_exact_p": (
                binomtest(min(gain, loss), discordant, 0.5).pvalue
                if discordant
                else 1.0
            ),
            "both_feasible_hpwl_cases": len(hpwl),
            "fallback_hpwl_relative_mean": (
                float(np.mean(hpwl)) if hpwl else None
            ),
        }
    variants = {}
    for variant in ("hard", "soft"):
        variants[variant] = {
            arm: {
                "with_fallback": summarize(fallback, arm, variant),
                "without_fallback": summarize(no_fallback, arm, variant),
            }
            for arm in ("candidate", "control")
        }
    return {
        "version": "chip-outline-fallback-effect-v1",
        "status": "structural_mechanism_diagnostic",
        "draws_per_arm": fallback["draws"],
        "overall": output,
        "variants": variants,
        "claim_boundary": (
            "This isolates the U3 fallback on the same structural cohort and "
            "does not establish routed-PPA or performance superiority."
        ),
    }


def main() -> None:
    args = parse_args()
    result = analyze(load(args.fallback), load(args.no_fallback))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
