#!/usr/bin/env python3
"""Aggregate matched chip-flow arms and apply the preregistered C1/C2 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("dense_no_ot", "dense_ot", "sparse16_no_ot", "sparse16_ot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def relative_delta(candidate: float, control: float) -> float:
    if control == 0.0:
        return 0.0
    return (candidate - control) / control


def main() -> None:
    args = parse_args()
    reports = {
        arm: json.loads((args.input_dir / f"{arm}.json").read_text())
        for arm in ARMS
    }
    curves = {
        arm: {row["step"]: row for row in report["history"]}
        for arm, report in reports.items()
    }
    common_steps = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    matched = []
    for step in common_steps:
        matched.append(
            {
                "step": step,
                **{
                    f"{arm}_total": curves[arm][step]["total"]
                    for arm in ARMS
                },
                **{
                    f"{arm}_overlap": curves[arm][step]["overlap"]
                    for arm in ARMS
                },
            }
        )

    final = {
        arm: reports[arm].get("final_dev") for arm in ARMS
    }
    dense_delta = relative_delta(
        final["dense_ot"]["total"],
        final["dense_no_ot"]["total"],
    )
    sparse_delta_cost = relative_delta(
        final["sparse16_ot"]["total"],
        final["sparse16_no_ot"]["total"],
    )
    dense_overlap_delta = relative_delta(
        final["dense_ot"]["overlap"],
        final["dense_no_ot"]["overlap"],
    )
    steps = {
        arm: reports[arm].get("steps_to_threshold") for arm in ARMS
    }
    dense_step_improvement = None
    if steps["dense_no_ot"] and steps["dense_ot"]:
        dense_step_improvement = (
            steps["dense_no_ot"] - steps["dense_ot"]
        ) / steps["dense_no_ot"]

    c1_existence = (
        final["dense_ot"]["total"] < final["dense_no_ot"]["total"]
        and final["dense_ot"]["overlap"] < final["dense_no_ot"]["overlap"]
    )
    c1_practical = dense_delta <= -0.10 or (
        dense_step_improvement is not None and dense_step_improvement >= 0.25
    )
    c2_neutral = (
        abs(
            relative_delta(
                final["sparse16_no_ot"]["total"],
                final["dense_no_ot"]["total"],
            )
        )
        < 0.05
    )
    summary = {
        "status": "complete"
        if all(report.get("status") == "complete" for report in reports.values())
        else "running",
        "common_curve_steps": common_steps,
        "matched_curves": matched,
        "final_dev": final,
        "thresholds": reports["dense_no_ot"].get("thresholds"),
        "steps_to_threshold": steps,
        "contrasts": {
            "dense_ot_total_delta": dense_delta,
            "dense_ot_overlap_delta": dense_overlap_delta,
            "sparse16_ot_total_delta": sparse_delta_cost,
            "dense_ot_step_improvement": dense_step_improvement,
        },
        "gates": {
            "C1_existence": "pass" if c1_existence else "fail",
            "C1_practical": "pass" if c1_practical else "fail",
            "C2_sparse_parity": "pass" if c2_neutral else "fail",
            "C2_compute_saving": "not-supported: top-k masks dense scores",
        },
        "claim_boundary": (
            "Passing C1 does not justify claiming Trinity SOTA. "
            "C2 cannot claim FLOP savings unless the attention kernel is changed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["contrasts"], indent=2))
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()
