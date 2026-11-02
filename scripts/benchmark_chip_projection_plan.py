#!/usr/bin/env python3
"""Benchmark reusable fixed/MIB projection plans against per-step recomputation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.hard_constraint_projector import (
    ProjectionOutput,
    _boundary_violation_count,
    _mib_shape_violations,
    _overlap_pairs,
    _project_boundary,
    _separate_overlaps,
    compiled_project_hard_constraints_latent,
    project_hard_constraints,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projections import project_conditioned_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_projection_plan_benchmark.json",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device, warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(trials):
        started = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1.0e3)
    return float(np.median(samples))


def legacy_projector(
    latent: torch.Tensor,
    target: torch.Tensor,
    constraints: torch.Tensor,
    areas: torch.Tensor,
    iterations: int,
) -> ProjectionOutput:
    """Recompute the conditioned projection exactly as the prior implementation."""

    projected = latent.clone()
    anchored = constraints[..., :2].ne(0).any(dim=-1)
    movable = ~anchored
    scale = areas.sum(dim=1).sqrt().clamp_min(1.0e-6)
    for _ in range(iterations):
        projected = project_conditioned_latent(projected, target, constraints)
        projected = _project_boundary(projected, constraints, areas)
        boxes = decode_boxes_batch(projected, areas)
        moved = _separate_overlaps(boxes, movable, constraints)
        projected[..., 0] += (moved[..., 0] - boxes[..., 0]) / scale[:, None]
        projected[..., 1] += (moved[..., 1] - boxes[..., 1]) / scale[:, None]
    projected = project_conditioned_latent(projected, target, constraints)
    projected = _project_boundary(projected, constraints, areas)
    boxes = decode_boxes_batch(projected, areas)
    target_boxes = decode_boxes_batch(target, areas)
    soft = ~constraints[..., :2].ne(0).any(dim=-1)
    fixed = constraints[..., 0].ne(0)
    preplaced = constraints[..., 1].ne(0)
    area_error = (boxes[..., 2] * boxes[..., 3] - areas).abs()[soft]
    fixed_error = (boxes[..., 2:] - target_boxes[..., 2:]).abs()[fixed]
    preplaced_error = (boxes - target_boxes).abs()[preplaced]
    return ProjectionOutput(
        latent=projected,
        boxes=boxes,
        remaining_overlap_pairs=_overlap_pairs(boxes),
        boundary_violations=_boundary_violation_count(boxes, constraints),
        mib_shape_violations=_mib_shape_violations(boxes, constraints),
        max_area_error=float(area_error.max()) if area_error.numel() else 0.0,
        fixed_shape_error=(
            float(fixed_error.max()) if fixed_error.numel() else 0.0
        ),
        preplaced_max_error=(
            float(preplaced_error.max()) if preplaced_error.numel() else 0.0
        ),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    store = PackedCaseStore(args.cache)
    indices = store.indices(args.block_count)[: args.batch_size]
    batch = store.batch(indices, device=device)
    latent = batch["target_latent"].clone()
    target = batch["target_latent"]
    constraints = batch["constraints"]
    areas = batch["areas"]

    with torch.no_grad():
        legacy_latent = legacy_projector(
            latent,
            target,
            constraints,
            areas,
            args.iterations,
        ).latent
        planned_latent = project_hard_constraints(
            latent,
            target,
            constraints,
            areas,
            iterations=args.iterations,
        ).latent
        legacy_ms = timed(
            lambda: legacy_projector(
                latent,
                target,
                constraints,
                areas,
                args.iterations,
            ),
            device,
            args.warmup,
            args.trials,
        )
        planned_ms = timed(
            lambda: project_hard_constraints(
                latent,
                target,
                constraints,
                areas,
                iterations=args.iterations,
            ),
            device,
            args.warmup,
            args.trials,
        )

    max_difference = float((legacy_latent - planned_latent).abs().max())
    compile_report = {}
    if args.compile:
        compiled = lambda: compiled_project_hard_constraints_latent(
            latent,
            target,
            constraints,
            areas,
            iterations=args.iterations,
            mode=args.compile_mode,
        )
        compiled_latent = compiled().clone()
        synchronize(device)
        compiled_ms = timed(
            compiled,
            device,
            args.warmup,
            args.trials,
        )
        compile_report = {
            "mode": args.compile_mode,
            "compiled_ms": compiled_ms,
            "speedup_vs_prepared": planned_ms / compiled_ms,
            "speedup_vs_legacy": legacy_ms / compiled_ms,
            "max_abs_difference_vs_prepared": float(
                (compiled_latent - planned_latent).abs().max()
            ),
        }
    forward_equal = torch.allclose(
        legacy_latent,
        planned_latent,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    compile_equal = (
        compile_report.get("max_abs_difference_vs_prepared", 0.0) <= 1.0e-5
        if compile_report
        else True
    )
    report = {
        "version": "chip-v46-projection-plan-benchmark-v1",
        "status": "pass" if forward_equal and compile_equal else "fail",
        "device": str(device),
        "block_count": args.block_count,
        "batch_size": len(indices),
        "iterations": args.iterations,
        "legacy_ms": legacy_ms,
        "prepared_plan_ms": planned_ms,
        "speedup": legacy_ms / planned_ms,
        "forward_max_abs_difference": max_difference,
        "compile": compile_report or None,
        "gradient_equivalence": (
            "Covered by tests/test_chip_projections.py for the reusable "
            "conditioned-projection plan."
        ),
        "reused_operator": (
            "prepare_conditioned_projection outside the project_hard_constraints loop"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
