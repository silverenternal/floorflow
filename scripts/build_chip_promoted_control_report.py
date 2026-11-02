#!/usr/bin/env python3
"""Compare full typed hierarchy with promoted-length structural controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_hpwl_gap",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["nfe4_refine400"]


def main() -> None:
    cases = {
        "full": load(
            ROOT
            / "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
            "reproducible_v1/candidate_projector8_primary_cases.json"
        ),
        "no_structural": load(
            ROOT
            / "outputs/chip_hierarchy/v9_promoted_controls/"
            "no_structural_primary_cases.json"
        ),
        "no_hierarchy": load(
            ROOT
            / "outputs/chip_hierarchy/v9_promoted_controls/"
            "no_hierarchy_primary_cases.json"
        ),
    }
    means = {
        arm: {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        }
        for arm, rows in cases.items()
    }
    comparisons = {}
    for control_index, arm in enumerate(("no_structural", "no_hierarchy")):
        comparisons[arm] = {
            metric: {
                "full_mean": means["full"][metric],
                "control_mean": means[arm][metric],
                "relative_delta": (
                    means["full"][metric] / means[arm][metric] - 1.0
                ),
                "paired": paired_delta(
                    cases["full"],
                    cases[arm],
                    metric,
                    iterations=4000,
                    seed=20261400 + control_index * 20 + index,
                    lower_is_better=True,
                ),
            }
            for index, metric in enumerate(METRICS)
        }
    checks = {
        "full_beats_both_controls_on_b2b": all(
            means["full"]["legalized_b2b_hpwl_gap"]
            < means[arm]["legalized_b2b_hpwl_gap"]
            for arm in ("no_structural", "no_hierarchy")
        ),
        "full_beats_both_controls_on_area": all(
            means["full"]["legalized_area_gap"]
            < means[arm]["legalized_area_gap"]
            for arm in ("no_structural", "no_hierarchy")
        ),
        "full_beats_no_structural_grouping": (
            means["full"]["legalized_grouping"]
            < means["no_structural"]["legalized_grouping"]
        ),
        "all_legalized_feasible_fraction_one": all(
            all(bool(row["legalized_feasible"]) for row in rows)
            for rows in cases.values()
        ),
    }
    report = {
        "status": (
            "pass_with_reported_tradeoffs"
            if all(checks.values())
            else "reject_promoted_mechanism"
        ),
        "scope": "seed-19 20k NFE4+400 projector8 promoted controls",
        "arm_means": means,
        "full_vs_controls": comparisons,
        "checks": checks,
        "interpretation": {
            "typed_structure": (
                "Full typed conditioning beats no-structural and no-hierarchy "
                "on area and b2b structural endpoints."
            ),
            "hierarchy_boundary": (
                "No-hierarchy has lower grouping error, so hierarchy is not "
                "claimed as a grouping improvement."
            ),
            "main_metric_tradeoff": (
                "No-structural has lower mean cost and V_rel; full typed is "
                "retained only for the predeclared structural claim."
            ),
        },
    }
    output = (
        ROOT
        / "outputs/reports/chip_typed_hierarchy_promoted_controls_v1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
