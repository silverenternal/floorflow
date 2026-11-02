#!/usr/bin/env python3
"""Audit the FloorSet-compatible hard-constraint projector design on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.data import load_validation_case
from trinity.floorplan.legalize.scale_pack import (
    last_info,
    scale_pack_legalize,
)
from trinity.floorplan.scoring import validate
from trinity.floorplan.types import Placement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=[21, 60, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_projector_design_audit_v1.json",
    )
    return parser.parse_args()


def stress_placement(instance) -> Placement:
    """Overlap body blocks while preserving known targets and exact block area."""

    gt = np.asarray(instance.gt_positions, dtype=np.float64)
    target = np.asarray(instance.target_positions, dtype=np.float64)
    areas = np.asarray(instance.area_targets, dtype=np.float64)
    centers = gt[:, :2] + gt[:, 2:] / 2.0
    center = centers.mean(axis=0)
    boxes = np.zeros((instance.block_count, 4), dtype=np.float64)
    for index, area in enumerate(areas):
        if instance.is_preplaced[index] and np.all(target[index] >= 0):
            boxes[index] = target[index]
            continue
        if instance.is_fixed[index] and target[index, 2] >= 0:
            width = target[index, 2]
            height = area / width
        else:
            width = np.sqrt(area)
            height = width
        boxes[index] = [
            center[0] - width / 2.0,
            center[1] - height / 2.0,
            width,
            height,
        ]
    return Placement(boxes, instance)


def audit_case(blocks: int) -> dict:
    instance = load_validation_case(blocks)
    before = stress_placement(instance)
    started = time.perf_counter()
    legalized = scale_pack_legalize(before)
    elapsed = time.perf_counter() - started
    validation = validate(legalized, "full_fast")
    score = validation.score
    feasibility = score.feasibility
    soft = score.soft
    return {
        "blocks": blocks,
        "seconds": elapsed,
        "scale_pack": last_info(),
        "feasible": validation.feasible,
        "hard_violations": {
            "overlap_pairs": feasibility.overlap_count,
            "area": len(feasibility.area_violations),
            "fixed": len(feasibility.fixed_violations),
            "preplaced": len(feasibility.preplaced_violations),
            "aspect": len(feasibility.aspect_violations),
        },
        "soft_violations": {
            "grouping": int(soft.grouping) if soft else 0,
            "mib": int(soft.mib) if soft else 0,
            "boundary": int(soft.boundary) if soft else 0,
        },
        "cost": float(score.cost),
        "v_rel": float(score.v_rel),
    }


def main() -> None:
    args = parse_args()
    rows = [audit_case(blocks) for blocks in args.blocks]
    checks = {
        "all_stress_layouts_feasible": all(row["feasible"] for row in rows),
        "no_hard_violations": all(
            all(value == 0 for value in row["hard_violations"].values())
            for row in rows
        ),
        "scale_pack_reports_iterations": all(
            "solves" in row["scale_pack"] for row in rows
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "redesign",
        "scope": "CPU stress audit for per-step and final hard-constraint projection",
        "rows": rows,
        "selected_design": {
            "final_oracle": "existing scale_pack",
            "per_step_projection": [
                "preserve fixed and preplaced boxes",
                "resolve overlap with a batched candidate/grid or continuous displacement pass",
                "project MIB members to one shared rho",
                "project boundary-coded blocks onto required bbox edges",
                "restore exact area and finite rho"
            ],
            "velocity_feedback": (
                "After projection z_proj, expose a residual correction toward "
                "z_proj in the ODE update and log its gate/causal effect."
            ),
            "flowplace_reuse": {
                "reusable": [
                    "grid candidate nearest-displacement overlap legalization",
                    "fixed/port immobility mask",
                    "padding between macros"
                ],
                "not_enough": [
                    "no FloorSet MIB shared-shape constraint",
                    "no FloorSet cluster abutment constraint",
                    "no boundary-code edge constraint",
                    "not the final exact-area FloorSet scorer"
                ]
            },
            "why_not_full_scale_pack_per_step": (
                "The LP ladder is a strong final legality oracle but too expensive "
                "and non-differentiable to run after every ODE step by default."
            ),
            "final_gate": (
                "Every rollout still ends with scale_pack before official scoring."
            ),
        },
        "checks": checks,
        "reuse_sources": [
            "third_party/Trinity/src/trinity/floorplan/legalize/scale_pack.py",
            "third_party/Trinity/src/trinity/floorplan/scoring/feasibility.py",
            "src/floorflow/chip_flow/projections.py",
            "https://github.com/lamda-bbo/flowplace/blob/main/flow_matching/hard_constraint_legalizer.py"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
