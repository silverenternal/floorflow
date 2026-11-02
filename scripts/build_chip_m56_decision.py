#!/usr/bin/env python3
"""Decide whether M56 legal-parent reflow advances from the dev screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

METRICS = (
    "cost",
    "v_rel",
    "area_gap",
    "b2b_hpwl_gap",
    "p2b_hpwl_gap",
    "grouping",
    "boundary",
    "rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen",
        type=Path,
        default=ROOT / "outputs/reports/chip_m56_single_parent_equal_budget_dev32_v1.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/reports/chip_m48_portfolio_dev32_postfix_v1.json",
    )
    parser.add_argument(
        "--m75",
        type=Path,
        default=ROOT / "outputs/reports/chip_m75_seed_diverse_guarded_dev32_v1.json",
    )
    parser.add_argument("--trinity", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m56_decision_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def selected_row(row: dict) -> dict:
    legalized = row["selected"]["stages"]["legalized"]
    return {
        "index": int(row["index"]),
        **{key: float(legalized[key]) for key in METRICS},
        "feasible": bool(legalized["feasible"]),
    }


def reference_row(row: dict) -> dict:
    legalized = row["selected"]["stages"]["legalized"]
    return {
        "index": int(row["index"]),
        **{key: float(legalized[key]) for key in METRICS},
        "feasible": bool(legalized["feasible"]),
    }


def main() -> None:
    args = parse_args()
    screen = load(args.screen)
    reference = load(args.reference)
    candidate_rows = sorted(
        [selected_row(row) for row in screen["rows"]],
        key=lambda row: row["index"],
    )
    reference_rows = sorted(
        [reference_row(row) for row in reference["rows"]],
        key=lambda row: row["index"],
    )
    if [row["index"] for row in candidate_rows] != [
        row["index"] for row in reference_rows
    ]:
        raise ValueError("M56 and reference cases do not align")
    paired = {
        key: paired_delta(
            candidate_rows,
            reference_rows,
            key,
            iterations=10000,
            seed=5600 + offset,
            lower_is_better=True,
        )
        for offset, key in enumerate(METRICS)
    }
    m75 = load(args.m75)
    m75_rows = sorted(m75["selected_rows"], key=lambda row: int(row["index"]))
    paired_to_m75 = {
        key: paired_delta(
            candidate_rows,
            m75_rows,
            key,
            iterations=10000,
            seed=5700 + offset,
            lower_is_better=True,
        )
        for offset, key in enumerate(METRICS)
    }
    paired_to_trinity = None
    if args.trinity is not None:
        trinity_payload = load(args.trinity)
        trinity_rows = sorted(
            [
                {
                    "index": int(row["index"]),
                    "cost": float(row["legalized_hard_cost"]),
                    "v_rel": float(row["legalized_v_rel"]),
                    "area_gap": float(row["legalized_area_gap"]),
                    "b2b_hpwl_gap": float(row["legalized_b2b_hpwl_gap"]),
                    "p2b_hpwl_gap": float(row["legalized_p2b_hpwl_gap"]),
                    "grouping": float(row["legalized_grouping"]),
                    "boundary": float(row["legalized_boundary"]),
                    "rho_mae": float(row["legalized_rho_mae"]),
                }
                for row in trinity_payload["rows"]
            ],
            key=lambda row: row["index"],
        )
        paired_to_trinity = {
            key: paired_delta(
                candidate_rows,
                trinity_rows,
                key,
                iterations=10000,
                seed=5800 + offset,
                lower_is_better=True,
            )
            for offset, key in enumerate(METRICS)
        }
    locality = [
        float(row["reflow_diagnostics"]["noncritical_displacement_mean"])
        for row in screen["rows"]
    ]
    checks = {
        "feasibility_one": all(row["feasible"] for row in candidate_rows),
        "no_solver_escalation": not screen["summary"]["solver_escalations"],
        "cost_improves": paired["cost"]["relative_delta"] < 0.0,
        "v_rel_non_regresses": paired["v_rel"]["relative_delta"] <= 0.0,
        "noncritical_locality_zero": all(value <= 1.0e-9 for value in locality),
    }
    report = {
        "version": "chip-m56-decision-v1",
        "status": "pass" if all(checks.values()) else "reject",
        "screen": str(args.screen),
        "reference": str(args.reference),
        "cases": len(candidate_rows),
        "checks": checks,
        "relative": {
            key: value["relative_delta"] for key, value in paired.items()
        },
        "paired": paired,
        "paired_to_m75": paired_to_m75,
        "paired_to_trinity": paired_to_trinity,
        "selection_counts": screen["summary"]["selection_counts"],
        "reflow_selected_cases": sum(
            row["selection"] == "reflow" for row in screen["rows"]
        ),
        "max_noncritical_displacement": max(locality, default=0.0),
        "budget": screen["budget"],
        "budget_ladder": {
            "m48_seed19_portfolio": {
                "nfe": 2,
                "refiner_steps": 3200,
                "legalizer_calls": 3,
                "cost": sum(row["cost"] for row in reference_rows) / len(reference_rows),
                "v_rel": sum(row["v_rel"] for row in reference_rows) / len(reference_rows),
            },
            "m56_single_parent_reflow": {
                **screen["budget"],
                "cost": sum(row["cost"] for row in candidate_rows) / len(candidate_rows),
                "v_rel": sum(row["v_rel"] for row in candidate_rows) / len(candidate_rows),
            },
            "m75_seed_diverse": {
                "nfe": 14,
                "refiner_steps": 16000,
                "legalizer_calls": 15,
                "cost": sum(row["cost"] for row in m75_rows) / len(m75_rows),
                "v_rel": sum(row["v_rel"] for row in m75_rows) / len(m75_rows),
            },
        },
        "claim_boundary": (
            "Frozen dev32 mechanism screen only. M56 is a lower-refiner "
            "candidate, not a replacement for the higher-quality M75 finalist. "
            "Official-100 and external routed PPA are not implied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks, "relative": report["relative"]}, indent=2))


if __name__ == "__main__":
    main()
