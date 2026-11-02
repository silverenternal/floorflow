#!/usr/bin/env python3
"""Stratify the rejected M70 Official-100 result by block complexity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

KEY = "nfe4_refine400"
CANDIDATE = (
    "outputs/reports/chip_m70_seed21_candidate_official100_ref400_v1.json"
)
CONTROL = "outputs/reports/chip_m70_seed21_control_official100_ref400_v1.json"
TRINITY = "outputs/reports/chip_trinity_d192_nfe4_ref400_official100_v1.json"
STRATA = (
    ("21-40", 21, 40),
    ("41-60", 41, 60),
    ("61-80", 61, 80),
    ("81-100", 81, 100),
    ("101-120", 101, 120),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m70_official100_strata_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load_cases(path: str, *, report: bool = False) -> list[dict]:
    path_obj = ROOT / path
    if report:
        return json.loads(path_obj.read_text(encoding="utf-8"))["rows"]
    cases = path_obj.with_name(f"{path_obj.stem}_cases.json")
    return json.loads(cases.read_text(encoding="utf-8"))[KEY]


def subset(rows: list[dict], low: int, high: int) -> list[dict]:
    return [row for row in rows if low <= int(row["blocks"]) <= high]


def relative(current: list[dict], baseline: list[dict], metric: str) -> float:
    current_mean = sum(float(row[metric]) for row in current) / len(current)
    baseline_mean = sum(float(row[metric]) for row in baseline) / len(baseline)
    return current_mean / baseline_mean - 1.0


def compare(
    current: list[dict],
    baseline: list[dict],
    *,
    iterations: int,
) -> dict:
    result = {}
    for offset, metric in enumerate(
        ("legalized_hard_cost", "legalized_v_rel")
    ):
        result[metric] = paired_delta(
            current,
            baseline,
            metric,
            iterations=iterations,
            seed=700 + offset,
            lower_is_better=True,
        )
    return result


def main() -> None:
    args = parse_args()
    candidate = load_cases(CANDIDATE)
    control = load_cases(CONTROL)
    trinity = load_cases(TRINITY, report=True)
    strata = {}
    for name, low, high in STRATA:
        candidate_rows = subset(candidate, low, high)
        control_rows = subset(control, low, high)
        trinity_rows = subset(trinity, low, high)
        strata[name] = {
            "cases": len(candidate_rows),
            "candidate_vs_control": {
                "relative": {
                    metric: relative(candidate_rows, control_rows, metric)
                    for metric in ("legalized_hard_cost", "legalized_v_rel")
                },
                "paired": compare(
                    candidate_rows,
                    control_rows,
                    iterations=args.bootstrap,
                ),
            },
            "candidate_vs_trinity": {
                "relative": {
                    metric: relative(candidate_rows, trinity_rows, metric)
                    for metric in ("legalized_hard_cost", "legalized_v_rel")
                },
                "paired": compare(
                    candidate_rows,
                    trinity_rows,
                    iterations=args.bootstrap,
                ),
            },
        }
    report = {
        "version": "chip-m70-official100-strata-v1",
        "status": "complete",
        "candidate": CANDIDATE,
        "control": CONTROL,
        "trinity": TRINITY,
        "strata": strata,
        "boundary": (
            "Post-rejection diagnostic only. It explains where M64 failed and "
            "cannot convert the rejected candidate into a promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "cases": row["cases"],
                    "cost_vs_control": row["candidate_vs_control"]["relative"][
                        "legalized_hard_cost"
                    ],
                    "v_rel_vs_control": row["candidate_vs_control"]["relative"][
                        "legalized_v_rel"
                    ],
                }
                for name, row in strata.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
