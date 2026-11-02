#!/usr/bin/env python3
"""CPU audit of per-step projection plus final scale_pack legality."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.data import load_validation_case
from trinity.floorplan.legalize.scale_pack import (
    last_info,
    scale_pack_legalize,
)
from trinity.floorplan.scoring import validate
from trinity.floorplan.types import Placement

from floorflow.chip_flow.hard_constraint_projector import (
    _boundary_violation_count,
    _mib_shape_violations,
    _overlap_pairs,
    project_hard_constraints,
)
from floorflow.chip_flow.model import decode_boxes_batch, encode_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=[21, 60, 120])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_projector_cpu_v1.json",
    )
    return parser.parse_args()


def stress_latents(instance) -> tuple[torch.Tensor, torch.Tensor]:
    target_boxes = torch.from_numpy(instance.gt_positions).float()
    areas = torch.from_numpy(instance.area_targets).float()[None]
    target = encode_latents(target_boxes[None], areas)
    source = target.clone()
    source[..., 0:2] = 0.5
    source[..., 2] = 0.0
    return source, target


def audit(blocks: int) -> dict:
    instance = load_validation_case(blocks)
    source, target = stress_latents(instance)
    areas = torch.from_numpy(instance.area_targets).float()[None]
    constraints = torch.from_numpy(instance.constraints).float()[None]
    started = time.perf_counter()
    source_boxes = decode_boxes_batch(source, areas)
    projected = project_hard_constraints(
        source,
        target,
        constraints,
        areas,
        iterations=16,
    )
    projection_seconds = time.perf_counter() - started
    final_candidate = scale_pack_legalize(
        Placement(projected.boxes[0].numpy(), instance)
    )
    validation = validate(final_candidate, "full_fast")
    feasibility = validation.score.feasibility
    return {
        "blocks": blocks,
        "projection_seconds": projection_seconds,
        "source_residuals": {
            "overlap_pairs": _overlap_pairs(source_boxes),
            "boundary_violations": _boundary_violation_count(
                source_boxes,
                constraints,
            ),
            "mib_shape_violations": _mib_shape_violations(
                source_boxes,
                constraints,
            ),
        },
        "fast_projection": {
            "remaining_overlap_pairs": projected.remaining_overlap_pairs,
            "boundary_violations": projected.boundary_violations,
            "mib_shape_violations": projected.mib_shape_violations,
            "max_area_error": projected.max_area_error,
            "fixed_shape_error": projected.fixed_shape_error,
            "preplaced_max_error": projected.preplaced_max_error,
        },
        "final_scale_pack": {
            "feasible": validation.feasible,
            "overlap_pairs": feasibility.overlap_count,
            "area_violations": len(feasibility.area_violations),
            "fixed_violations": len(feasibility.fixed_violations),
            "preplaced_violations": len(feasibility.preplaced_violations),
            "aspect_violations": len(feasibility.aspect_violations),
            "info": last_info(),
        },
    }


def main() -> None:
    args = parse_args()
    rows = [audit(blocks) for blocks in args.blocks]
    checks = {
        "fast_projection_exact_area": all(
            row["fast_projection"]["max_area_error"] < 1.0e-4
            for row in rows
        ),
        "fast_projection_reduces_overlap": all(
            row["fast_projection"]["remaining_overlap_pairs"]
            < row["source_residuals"]["overlap_pairs"]
            for row in rows
        ),
        "fast_projection_fixed_shape": all(
            row["fast_projection"]["fixed_shape_error"] < 1.0e-5
            for row in rows
        ),
        "fast_projection_preplaced": all(
            row["fast_projection"]["preplaced_max_error"] < 1.0e-5
            for row in rows
        ),
        "final_scale_pack_feasible": all(
            row["final_scale_pack"]["feasible"] for row in rows
        ),
        "final_zero_hard_violations": all(
            all(
                value == 0
                for key, value in row["final_scale_pack"].items()
                if key.endswith("violations") or key == "overlap_pairs"
            )
            for row in rows
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "redesign",
        "scope": "CPU fast projection and final scale_pack legality audit",
        "rows": rows,
        "checks": checks,
        "policy": {
            "per_step": "fast approximate projection + velocity correction",
            "final": "scale_pack remains the official legality oracle",
            "target_use": "only fixed-shape/preplaced rows may read condition targets",
            "guarantee": "intermediate projection is approximate; final hard legality is exact before scoring"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
