#!/usr/bin/env python3
"""Compare candidate and matched-control refiner saturation curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_m1_candidate_saturation_screen.json",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_m1_control_saturation_screen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_v45_refiner_saturation_screen.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float:
    return float((candidate - control) / control) if control else 0.0


def main() -> None:
    args = parse_args()
    candidate = load(args.candidate)
    control = load(args.control)
    candidate_steps = sorted(
        int(key.rsplit("refine", 1)[1])
        for key in candidate["results"]
        if key.startswith("nfe4_refine")
    )
    control_steps = sorted(
        int(key.rsplit("refine", 1)[1])
        for key in control["results"]
        if key.startswith("nfe4_refine")
    )
    if candidate_steps != control_steps:
        raise ValueError("candidate and control refiner-step grids differ")

    rows = []
    previous_candidate = None
    previous_control = None
    for step in candidate_steps:
        key = f"nfe4_refine{step}"
        left = candidate["results"][key]
        right = control["results"][key]
        row = {
            "step": step,
            "candidate_cost": left["legalized_hard_cost"],
            "control_cost": right["legalized_hard_cost"],
            "cost_relative": relative(
                left["legalized_hard_cost"],
                right["legalized_hard_cost"],
            ),
            "candidate_v_rel": left["legalized_v_rel"],
            "control_v_rel": right["legalized_v_rel"],
            "v_rel_relative": relative(
                left["legalized_v_rel"],
                right["legalized_v_rel"],
            ),
            "candidate_feasible_fraction": left["legalized_feasible_fraction"],
            "control_feasible_fraction": right["legalized_feasible_fraction"],
            "candidate_update_rms": left.get(
                "latent_update_rms_per_coordinate"
            ),
            "control_update_rms": right.get(
                "latent_update_rms_per_coordinate"
            ),
        }
        if previous_candidate is not None:
            row["candidate_cost_change_from_previous"] = (
                left["legalized_hard_cost"]
                - previous_candidate["legalized_hard_cost"]
            )
            row["candidate_v_rel_change_from_previous"] = (
                left["legalized_v_rel"]
                - previous_candidate["legalized_v_rel"]
            )
            row["control_cost_change_from_previous"] = (
                right["legalized_hard_cost"]
                - previous_control["legalized_hard_cost"]
            )
            row["control_v_rel_change_from_previous"] = (
                right["legalized_v_rel"]
                - previous_control["legalized_v_rel"]
            )
        rows.append(row)
        previous_candidate = left
        previous_control = right

    matched = next((row for row in rows if row["step"] == 1600), None)
    checks = {
        "matched_1600_present": matched is not None,
        "matched_1600_cost_win": bool(
            matched and matched["candidate_cost"] < matched["control_cost"]
        ),
        "matched_1600_v_rel_win": bool(
            matched and matched["candidate_v_rel"] < matched["control_v_rel"]
        ),
        "all_legal_feasible": all(
            row["candidate_feasible_fraction"] == 1.0
            and row["control_feasible_fraction"] == 1.0
            for row in rows
        ),
    }
    report = {
        "version": "chip-v45-refiner-saturation-v1",
        "status": (
            "matched_1600_lead"
            if checks["matched_1600_cost_win"]
            and checks["matched_1600_v_rel_win"]
            else "matched_1600_not_passed"
        ),
        "candidate_report": str(args.candidate.resolve().relative_to(ROOT)),
        "control_report": str(args.control.resolve().relative_to(ROOT)),
        "scope": "32-case deterministic block-count-stratified dev screen, seed 19",
        "rows": rows,
        "checks": checks,
        "next_action": (
            "Freeze at most two schedules for M2."
            if checks["matched_1600_cost_win"]
            and checks["matched_1600_v_rel_win"]
            else "Inspect plateaus; if both arms plateau, prioritize checkpoint and training work."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
