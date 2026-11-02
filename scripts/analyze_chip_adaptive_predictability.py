#!/usr/bin/env python3
"""Measure whether runtime observables can select refiner snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import total_mean

FEATURES = (
    "total",
    "structural",
    "overlap",
    "group",
    "mib",
    "boundary",
    "wl",
    "area",
    "trust_region_rms",
    "incremental_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_value(row: dict, name: str) -> float:
    if name == "structural":
        return sum(
            float(row[key])
            for key in ("overlap", "group", "mib", "boundary", "area")
        )
    return float(row[name])


def joined_cases(report: dict) -> dict[int, list[dict]]:
    observable = {
        (int(row["index"]), int(row["refiner_steps"])): row
        for row in report["rows"]
    }
    legalized = {
        (int(row["index"]), int(row["refiner_steps"])): row
        for row in report["legalized"]
    }
    if set(observable) != set(legalized):
        raise ValueError("observable and legalized rows do not align")
    grouped: dict[int, list[dict]] = {}
    for index, steps in sorted(legalized):
        grouped.setdefault(index, []).append(
            {
                "index": index,
                "refiner_steps": steps,
                "observable": observable[(index, steps)],
                "legalized": legalized[(index, steps)],
            }
        )
    return grouped


def summarize(rows: list[dict]) -> dict:
    costs = [
        float(row["legalized"]["legalized_hard_cost"])
        for row in rows
    ]
    blocks = [
        int(row["legalized"]["blocks"])
        for row in rows
    ]
    return {
        "cases": len(rows),
        "mean_refiner_steps": float(
            np.mean([int(row["refiner_steps"]) for row in rows])
        ),
        "legalized_hard_cost": total_mean(costs, blocks),
        "legalized_v_rel": float(
            np.mean(
                [
                    float(row["legalized"]["legalized_v_rel"])
                    for row in rows
                ]
            )
        ),
        "legalized_feasible_fraction": float(
            np.mean(
                [
                    bool(row["legalized"]["legalized_feasible"])
                    for row in rows
                ]
            )
        ),
    }


def fixed_summary(cases: dict[int, list[dict]], steps: int) -> dict:
    rows = []
    for case in cases.values():
        selected = [row for row in case if int(row["refiner_steps"]) == steps]
        if len(selected) != 1:
            raise ValueError(f"case {case[0]['index']} lacks fixed step {steps}")
        rows.append(selected[0])
    return {"steps": steps, **summarize(rows)}


def rank_policy_summary(
    cases: dict[int, list[dict]],
    feature: str,
    direction: str,
) -> dict:
    sign = 1.0 if direction == "min" else -1.0
    selected = []
    for case in cases.values():
        selected.append(
            min(
                case,
                key=lambda row: (
                    sign * feature_value(row["observable"], feature),
                    int(row["refiner_steps"]),
                ),
            )
        )
    return {"feature": feature, "direction": direction, **summarize(selected)}


def within_case_rank_consistency(cases: dict[int, list[dict]], feature: str) -> dict:
    correlations = []
    for case in cases.values():
        values = [
            feature_value(row["observable"], feature)
            for row in case
        ]
        costs = [
            float(row["legalized"]["legalized_hard_cost"])
            for row in case
        ]
        if np.ptp(values) == 0.0 or np.ptp(costs) == 0.0:
            continue
        correlation, _ = spearmanr(values, costs)
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    return {
        "cases": len(correlations),
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "median_spearman": float(np.median(correlations)) if correlations else None,
        "positive_fraction": (
            float(np.mean(np.asarray(correlations) > 0.0))
            if correlations
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    report = load(args.snapshots)
    cases = joined_cases(report)
    snapshots = sorted(
        {
            int(row["refiner_steps"])
            for case in cases.values()
            for row in case
        }
    )
    fixed = [fixed_summary(cases, steps) for steps in snapshots]
    oracle_rows = [
        min(
            case,
            key=lambda row: float(
                row["legalized"]["legalized_hard_cost"]
            ),
        )
        for case in cases.values()
    ]
    observable_rank = [
        rank_policy_summary(cases, feature, direction)
        for feature in FEATURES
        for direction in ("min", "max")
    ]
    consistency = {
        feature: within_case_rank_consistency(cases, feature)
        for feature in FEATURES
    }
    output = {
        "version": "chip-sota-adaptive-predictability-v1",
        "status": "retrospective_diagnostic_only",
        "snapshot_report": str(args.snapshots),
        "case_manifest": report["case_manifest"],
        "nfe": report["nfe"],
        "seed": report["seed"],
        "cases": len(cases),
        "snapshots": snapshots,
        "fixed": fixed,
        "cost_oracle": summarize(oracle_rows),
        "observable_rank": observable_rank,
        "within_case_rank_consistency": consistency,
        "claim_boundary": (
            "Oracle rows and all rank policies are retrospective diagnostics on "
            "the same cases. No selector is validated by this report."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "cases": output["cases"],
                "fixed": fixed,
                "cost_oracle": output["cost_oracle"],
            }
        )
    )


if __name__ == "__main__":
    main()
