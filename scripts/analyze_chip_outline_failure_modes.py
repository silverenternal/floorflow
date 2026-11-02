#!/usr/bin/env python3
"""Classify fixed-outline draw failures by feasibility component."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v46_bookshelf_outline_fallback_full22_16draw_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_outline_failure_modes_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_modes(report: dict, arm: str, variant: str | None = None) -> dict:
    counts = Counter()
    cases = [
        case
        for case in report["cases"]
        if variant is None or case["id"].split("/")[1].lower() == variant
    ]
    for case in cases:
        for draw in case[arm]["draws"]:
            key = (
                bool(draw["fits_outline"]),
                bool(draw["overlap_free"]),
                bool(draw["shapes_kept"]),
            )
            counts[key] += 1
    total = sum(counts.values())
    return {
        "draws": total,
        "fully_feasible": sum(
            value for key, value in counts.items() if all(key)
        ),
        "outline_miss_no_overlap": counts[(False, True, True)],
        "outline_and_overlap_fail": (
            counts[(False, False, True)] + counts[(False, False, False)]
        ),
        "outline_pass_overlap_fail": counts[(True, False, True)],
        "shape_violation": sum(
            value for key, value in counts.items() if not key[2]
        ),
        "raw_counts": {
            "|".join("1" if item else "0" for item in key): value
            for key, value in sorted(counts.items())
        },
    }


def analyze(report: dict) -> dict:
    return {
        "version": "chip-outline-failure-modes-v1",
        "status": "diagnostic_only",
        "input": "Official fixed-outline structural cohort",
        "overall": {
            arm: arm_modes(report, arm)
            for arm in ("candidate", "control")
        },
        "variants": {
            variant: {
                arm: arm_modes(report, arm, variant)
                for arm in ("candidate", "control")
            }
            for variant in ("hard", "soft")
        },
        "claim_boundary": (
            "Draw-level failure classification is structural diagnostic evidence; "
            "it does not rank feasible layouts or imply routed PPA."
        ),
    }


def main() -> None:
    args = parse_args()
    result = analyze(load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
