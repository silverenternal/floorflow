#!/usr/bin/env python3
"""Benchmark exact chip data, metric and assignment hot-path migration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore, prepare_training_store
from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints
from floorflow.chip_flow.projector_geometry import (
    boundary_violation_count,
    joint_conflict_energy,
    mib_shape_violation_count,
)
from floorflow.chip_flow.proposals import token_ot_reorder_batch
from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.chip_physics import group_penalty, group_penalty_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-count", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_gpu_hotpath_benchmark_v1.json",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    function,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1.0e3)
    return float(np.median(samples))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    source_store = PackedCaseStore(args.cache)
    indices = source_store.indices(args.block_count)[: args.batch_size]
    if indices.size == 0:
        raise ValueError(f"cache has no cases with {args.block_count} blocks")
    resident_store, status = prepare_training_store(
        source_store,
        device,
        enabled=True,
        max_gib=16.0,
    )
    if not status["resident"]:
        raise RuntimeError(f"GPU cache did not fit: {status}")

    cpu_batch = source_store.batch(indices)
    gpu_batch = resident_store.batch(indices)
    synchronize(device)

    data = {
        "sync_unpinned_ms": timed(
            lambda: source_store.batch(
                indices,
                device=device,
                pin_memory=False,
                non_blocking=False,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "pinned_async_ms": timed(
            lambda: source_store.batch(
                indices,
                device=device,
                pin_memory=True,
                non_blocking=True,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "resident_index_ms": timed(
            lambda: resident_store.batch(indices),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    data["pinned_speedup"] = (
        data["sync_unpinned_ms"] / data["pinned_async_ms"]
    )
    data["resident_speedup_vs_sync"] = (
        data["sync_unpinned_ms"] / data["resident_index_ms"]
    )

    cpu_boxes = cpu_batch["boxes"]
    cpu_areas = cpu_batch["areas"]
    cpu_constraints = cpu_batch["constraints"]

    def legacy_group() -> torch.Tensor:
        return torch.stack(
            [
                group_penalty(
                    cpu_boxes[index],
                    cpu_constraints[index, :, 3],
                    cpu_areas[index],
                )
                for index in range(cpu_boxes.shape[0])
            ]
        )

    expected_group = legacy_group()
    actual_group = group_penalty_batch(
        gpu_batch["boxes"],
        gpu_batch["constraints"][..., 3],
        gpu_batch["areas"],
        workers=4,
    )
    torch.testing.assert_close(actual_group.cpu(), expected_group)
    grouping = {
        "legacy_case_loop_ms": timed(
            legacy_group,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "batched_gpu_gap_ms": timed(
            lambda: group_penalty_batch(
                gpu_batch["boxes"],
                gpu_batch["constraints"][..., 3],
                gpu_batch["areas"],
                workers=4,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "max_abs_difference": float(
            (actual_group.cpu() - expected_group).abs().max()
        ),
    }
    grouping["speedup"] = (
        grouping["legacy_case_loop_ms"] / grouping["batched_gpu_gap_ms"]
    )

    cpu_latent = gpu_batch["target_latent"].cpu()
    gpu_latent = gpu_batch["target_latent"]
    diagnostics = {
        "boundary_cpu_ms": timed(
            lambda: boundary_violation_count(cpu_boxes, cpu_constraints),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "boundary_gpu_ms": timed(
            lambda: boundary_violation_count(
                gpu_batch["boxes"],
                gpu_batch["constraints"],
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "mib_cpu_ms": timed(
            lambda: mib_shape_violation_count(cpu_boxes, cpu_constraints),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "mib_gpu_ms": timed(
            lambda: mib_shape_violation_count(
                gpu_batch["boxes"],
                gpu_batch["constraints"],
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "joint_cpu_ms": timed(
            lambda: joint_conflict_energy(
                cpu_latent,
                cpu_constraints,
                cpu_areas,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "joint_gpu_ms": timed(
            lambda: joint_conflict_energy(
                gpu_latent,
                gpu_batch["constraints"],
                gpu_batch["areas"],
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    for name in ("boundary", "mib", "joint"):
        diagnostics[f"{name}_speedup"] = (
            diagnostics[f"{name}_cpu_ms"] / diagnostics[f"{name}_gpu_ms"]
        )

    projected_with_diagnostics = project_hard_constraints(
        gpu_latent,
        gpu_batch["target_latent"],
        gpu_batch["constraints"],
        gpu_batch["areas"],
        iterations=4,
        collect_diagnostics=True,
    )
    projected_without_diagnostics = project_hard_constraints(
        gpu_latent,
        gpu_batch["target_latent"],
        gpu_batch["constraints"],
        gpu_batch["areas"],
        iterations=4,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(
        projected_without_diagnostics.latent,
        projected_with_diagnostics.latent,
    )
    projector = {
        "diagnostics_ms": timed(
            lambda: project_hard_constraints(
                gpu_latent,
                gpu_batch["target_latent"],
                gpu_batch["constraints"],
                gpu_batch["areas"],
                iterations=4,
                collect_diagnostics=True,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "latent_only_ms": timed(
            lambda: project_hard_constraints(
                gpu_latent,
                gpu_batch["target_latent"],
                gpu_batch["constraints"],
                gpu_batch["areas"],
                iterations=4,
                collect_diagnostics=False,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "trajectory_max_abs_difference": 0.0,
    }
    projector["speedup"] = (
        projector["diagnostics_ms"] / projector["latent_only_ms"]
    )

    cpu_source = cpu_latent
    cpu_target = cpu_batch["target_latent"]
    exact = {
        "cpu_tensor_ms": timed(
            lambda: token_ot_reorder_batch(
                cpu_source,
                cpu_target,
                workers=4,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "gpu_tensor_to_exact_cpu_solver_ms": timed(
            lambda: token_ot_reorder_batch(
                gpu_latent,
                gpu_batch["target_latent"],
                workers=4,
            ),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    exact["transfer_penalty_ratio"] = (
        exact["gpu_tensor_to_exact_cpu_solver_ms"] / exact["cpu_tensor_ms"]
    )

    report = {
        "scope": "exact equivalence and wall-time diagnostics; no quality claim",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "block_count": int(args.block_count),
        "batch_size": len(indices),
        "gpu_data_cache": status,
        "data": data,
        "grouping_exact_mst": grouping,
        "projector_diagnostics": diagnostics,
        "projector_latent_only": projector,
        "token_ot_exact_solver": exact,
        "decision": {
            "data": "resident index when it fits; pinned async fallback",
            "grouping": "GPU gap matrices; SciPy exact MST stays on CPU",
            "projector": "all-count and group reductions on GPU",
            "token_ot": "SciPy Hungarian stays on CPU; no exact CUDA dependency",
        },
    }
    write_json_atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
