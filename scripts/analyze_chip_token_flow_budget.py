#!/usr/bin/env python3
"""Compute equal-quality NFE and refiner budgets from matched diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_token_flow_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--refiner-curve",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_refiner_curve_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_budget_curve_v1.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_curve(payload: dict) -> dict:
    output = {}
    for arm in ("candidate", "baseline"):
        output[arm] = {}
        for step in payload["steps"]:
            rows = [
                row
                for row in payload["rows"]
                if row["arm"] == arm and row["refiner_steps"] == step
            ]
            output[arm][str(step)] = {
                "cases": len(rows),
                "cost": float(
                    np.mean([row["legalized_hard_cost"] for row in rows])
                ),
                "v_rel": float(
                    np.mean([row["legalized_v_rel"] for row in rows])
                ),
            }
    return output


def first_step_at_quality(
    curve: dict,
    thresholds: dict,
    *,
    steps: list[int],
) -> int | None:
    for step in steps:
        value = curve[str(step)]
        if (
            value["cost"] <= thresholds["cost"]
            and value["v_rel"] <= thresholds["v_rel"]
        ):
            return step
    return None


def main() -> None:
    args = parse_args()
    mechanism = read(args.mechanism)
    refiner_payload = read(args.refiner_curve)
    refiner = summarize_curve(refiner_payload)
    baseline_400 = refiner["baseline"]["400"]
    thresholds = {
        "cost": baseline_400["cost"],
        "v_rel": baseline_400["v_rel"],
    }
    candidate_refiner_step = first_step_at_quality(
        refiner["candidate"],
        thresholds,
        steps=refiner_payload["steps"],
    )

    baseline_nfe8 = mechanism["nfe_curves"]["8"]["baseline_mean"]
    nfe_candidates = []
    for nfe, value in mechanism["nfe_curves"].items():
        candidate = value["candidate_mean"]
        if (
            candidate["cost"] <= baseline_nfe8["cost"]
            and candidate["v_rel"] <= baseline_nfe8["v_rel"]
        ):
            nfe_candidates.append(int(nfe))
    candidate_nfe = min(nfe_candidates) if nfe_candidates else None
    checks = {
        "refiner_budget_reduced": (
            candidate_refiner_step is not None
            and candidate_refiner_step < 400
        ),
        "nfe_budget_reduced": candidate_nfe is not None and candidate_nfe < 8,
    }
    report = {
        "status": "supported" if any(checks.values()) else "not_supported",
        "scope": "same-quality inference budget on matched official-100 diagnostics",
        "refiner": {
            "curve": refiner,
            "baseline_quality_threshold": thresholds,
            "candidate_minimum_steps": candidate_refiner_step,
            "saved_steps": (
                400 - candidate_refiner_step
                if candidate_refiner_step is not None
                else 0
            ),
        },
        "nfe": {
            "baseline_nfe8_raw": baseline_nfe8,
            "candidate_qualifying_nfes": nfe_candidates,
            "candidate_minimum_nfe": candidate_nfe,
            "saved_nfe_vs_8": 8 - candidate_nfe if candidate_nfe is not None else 0,
        },
        "checks": checks,
        "boundary": (
            "The refiner curve is seed 19 and is a controlled inference diagnostic; "
            "the primary three-seed endpoint remains NFE4 plus 400 steps."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
