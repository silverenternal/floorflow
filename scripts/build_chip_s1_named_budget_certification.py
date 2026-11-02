#!/usr/bin/env python3
"""Certify the M75 finalist under named chip inference budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_s1_named_budget_certification_v1.json",
    )
    parser.add_argument(
        "--pareto-output",
        type=Path,
        default=ROOT / "outputs/reports/chip_s1_equal_quality_pareto_v1.json",
    )
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean_rows(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def point(
    name: str,
    report: dict,
    *,
    rows: list[dict],
    trajectories: int,
    nfe: int,
    refiner_steps: int,
    legalizer_calls: int,
    wall_clock_seconds: float | None,
) -> dict:
    return {
        "name": name,
        "report": report.get("version"),
        "cost": mean_rows(rows, "cost"),
        "v_rel": mean_rows(rows, "v_rel"),
        "area_gap": mean_rows(rows, "area_gap"),
        "b2b_hpwl_gap": mean_rows(rows, "b2b_hpwl_gap"),
        "feasible_fraction": mean_rows(rows, "feasible"),
        "trajectories": trajectories,
        "nfe": nfe,
        "refiner_steps": refiner_steps,
        "legalizer_calls": legalizer_calls,
        "wall_clock_seconds": wall_clock_seconds,
    }


def main() -> None:
    args = parse_args()
    m48 = load("outputs/reports/chip_m48_portfolio_official100_postfix_seed19.json")
    m74 = load("outputs/reports/chip_m73_cross_checkpoint_portfolio_official100_v1.json")
    m75 = load("outputs/reports/chip_m75_seed_diverse_guarded_official100_v1.json")
    m75_decision = load("outputs/reports/chip_m75_decision_v1.json")
    lineage = load("outputs/reports/chip_architecture_lineage_audit_v1.json")
    trinity_reports = {
        19: load("outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_v1.json"),
        20: load("outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_seed20.json"),
        21: load("outputs/reports/chip_trinity_d192_nfe4_ref3200_official100_seed21.json"),
    }
    m48_rows = [
        {
            "cost": row["selected"]["stages"]["legalized"]["cost"],
            "v_rel": row["selected"]["stages"]["legalized"]["v_rel"],
            "area_gap": row["selected"]["stages"]["legalized"]["area_gap"],
            "b2b_hpwl_gap": row["selected"]["stages"]["legalized"]["b2b_hpwl_gap"],
            "feasible": row["selected"]["stages"]["legalized"]["feasible"],
        }
        for row in m48["rows"]
    ]
    candidate_points = [
        point(
            "m48_portfolio_seed19",
            m48,
            rows=m48_rows,
            trajectories=2,
            nfe=2,
            refiner_steps=3200,
            legalizer_calls=3,
            wall_clock_seconds=float(m48["summary"]["wall_clock_seconds"]),
        ),
        point(
            "m74_fixed_seed_cross_checkpoint",
            m74,
            rows=m74["selected_rows"],
            trajectories=6,
            nfe=10,
            refiner_steps=9600,
            legalizer_calls=9,
            wall_clock_seconds=None,
        ),
        point(
            "m75_seed_diverse_guarded",
            m75,
            rows=m75["selected_rows"],
            trajectories=10,
            nfe=14,
            refiner_steps=16000,
            legalizer_calls=15,
            wall_clock_seconds=None,
        ),
    ]
    anchor_points = {
        str(seed): {
            "cost": float(report["mean_cost"]),
            "v_rel": float(report["mean_v_rel"]),
            "area_gap": float(report["mean_area_gap"]),
            "b2b_hpwl_gap": sum(
                float(
                    row["legalized_b2b_hpwl_gap"]
                    if "legalized_b2b_hpwl_gap" in row
                    else row["b2b_hpwl_gap"]
                )
                for row in report["rows"]
            )
            / len(report["rows"]),
            "feasible_fraction": float(report["feasible_fraction"]),
            "trajectories": 1,
            "nfe": 4,
            "refiner_steps": 3200,
            "legalizer_calls": 1,
            "mean_seconds": float(report["mean_seconds"]),
        }
        for seed, report in trinity_reports.items()
    }
    cost_threshold = min(anchor["cost"] for anchor in anchor_points.values())
    v_rel_threshold = min(anchor["v_rel"] for anchor in anchor_points.values())
    all_anchor_cost_threshold = max(anchor["cost"] for anchor in anchor_points.values())
    all_anchor_v_rel_threshold = max(
        anchor["v_rel"] for anchor in anchor_points.values()
    )
    for candidate in candidate_points:
        candidate["beats_min_cost_threshold"] = candidate["cost"] <= cost_threshold
        candidate["beats_min_v_rel_threshold"] = (
            candidate["v_rel"] <= v_rel_threshold
        )
        candidate["beats_every_anchor_mean"] = (
            candidate["cost"] <= cost_threshold
            and candidate["v_rel"] <= v_rel_threshold
        )
    finalist = candidate_points[-1]
    equal_inference = {
        "supported": False,
        "reason": (
            "No retained finalist matches the one-trajectory NFE4/refiner3200 "
            "anchor budget. M74 fixed-seed cross-checkpoint cost was not stable."
        ),
    }
    equal_quality = {
        "supported_as_fewer_budget": False,
        "minimum_existing_candidate_name": "m75_seed_diverse_guarded",
        "minimum_existing_candidate_nfe": 14,
        "minimum_existing_candidate_refiner_steps": 16000,
        "minimum_existing_candidate_legalizer_calls": 15,
        "minimum_existing_candidate_meets_every_anchor_mean": True,
        "reason": (
            "M75 reaches the quality region but uses more NFE, refiner work "
            "and legalizer calls than the anchor. It is an equal-or-better "
            "quality point, not an equal-quality efficiency win."
        ),
    }
    absolute_budget = {
        "supported": bool(
            m75_decision["checks"]["pooled_cost_ci_excludes_zero"]
            and m75_decision["checks"]["pooled_v_rel_ci_excludes_zero"]
            and m75_decision["checks"]["candidate_feasibility_one"]
        ),
        "budget": m75["budget"],
        "cost_mean_anchor_wins": m75_decision["cost_mean_anchor_wins"],
        "v_rel_mean_anchor_wins": m75_decision["v_rel_mean_anchor_wins"],
    }
    guards = {
        "area_non_regression": (
            finalist["area_gap"]
            <= min(anchor["area_gap"] for anchor in anchor_points.values())
        ),
        "b2b_non_regression": (
            finalist["b2b_hpwl_gap"]
            <= min(anchor["b2b_hpwl_gap"] for anchor in anchor_points.values())
        ),
        "external_ppa": "not_run",
        "timing_power_drc": "not_run",
    }
    report = {
        "version": "chip-s1-named-budget-certification-v1",
        "status": (
            "pass_absolute_budget_only"
            if absolute_budget["supported"]
            else "fail"
        ),
        "finalist": "m75_seed_diverse_guarded",
        "policy": m75["policy"],
        "parameter_budget": {
            "candidate": int(lineage["parameter_budget"]["candidate"]),
            "trinity_d192": int(lineage["parameter_budget"]["trinity_d192"]),
            "candidate_relative": float(
                lineage["parameter_budget"]["candidate_relative"]
            ),
            "candidate_training_exposures": int(
                lineage["training_budget"]["candidate_exposures"]
            ),
            "trinity_nominal_training_exposures": int(
                lineage["training_budget"]["trinity_d192_nominal"]
            ),
        },
        "anchor_quality_thresholds": {
            "minimum_cost": cost_threshold,
            "minimum_v_rel": v_rel_threshold,
            "maximum_anchor_cost": all_anchor_cost_threshold,
            "maximum_anchor_v_rel": all_anchor_v_rel_threshold,
        },
        "budget_classes": {
            "equal_inference": equal_inference,
            "equal_quality": equal_quality,
            "absolute": absolute_budget,
        },
        "guards": guards,
        "candidate_points": candidate_points,
        "anchors": anchor_points,
        "wall_time_boundary": (
            "Candidate wall time is not reconstructed because source reports "
            "used different serial/sharded schedules. Count budgets and "
            "parameter/FLOP accounting are the release-grade fields."
        ),
        "claim_decision": (
            "Absolute-budget internal candidate only. Do not claim "
            "same-inference or equal-quality efficiency, SOTA over the "
            "flagship anchor, area dominance or external routed-PPA transfer."
        ),
    }
    pareto = {
        "version": "chip-s1-equal-quality-pareto-v1",
        "status": report["status"],
        "candidate_points": candidate_points,
        "anchors": anchor_points,
        "equal_quality_threshold": {
            "cost": cost_threshold,
            "v_rel": v_rel_threshold,
        },
        "reading": (
            "M75 is the only retained candidate point that is below every "
            "anchor mean threshold on both primaries, but it does so with "
            "greater search and refiner budget."
        ),
        "claim_boundary": report["claim_decision"],
    }
    for path, payload in (
        (args.output, report),
        (args.pareto_output, pareto),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
