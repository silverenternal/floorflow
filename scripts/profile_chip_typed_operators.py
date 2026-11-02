#!/usr/bin/env python3
"""Profile the production typed-hierarchy flow path on a real FloorSet batch."""

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
from floorflow.chip_flow.flow_adapter import build_chip_model
from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints
from floorflow.chip_flow.hierarchical_flow_runtime import (
    hierarchical_training_state,
    hierarchy_flow_loss,
    interpolate_hierarchy,
    sample_hierarchy_flow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--compile-projector", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_operator_profile.json",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    fn,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1.0e3)
    return float(np.median(samples))


def production_config() -> dict:
    return {
        "architecture": "typed_hierarchy",
        "hidden": 192,
        "depth": 12,
        "heads": 6,
        "typed_layers": 2,
        "typed_mode": "typed",
        "graph_pe_k": 2,
        "projector": True,
        "projector_iterations": 8,
    }


def training_step(
    model: torch.nn.Module,
    batch: dict,
    generator: torch.Generator,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    context = model.prepare_context(batch)
    source, target = hierarchical_training_state(batch, context, generator)
    time_value = torch.full(
        (batch["features"].shape[0],),
        0.5,
        device=batch["features"].device,
    )
    interpolant = interpolate_hierarchy(source, target, time_value)
    prediction = model(interpolant, time_value, context)
    loss, _ = hierarchy_flow_loss(
        prediction,
        target,
        huber_beta=0.0,
        global_weight=0.1,
        structural_weight=0.1,
        port_weight=0.1,
    )
    loss.backward()
    return loss.detach()


def kernel_profile(
    model: torch.nn.Module,
    batch: dict,
    generator: torch.Generator,
    device: torch.device,
) -> list[dict]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=device.type == "cuda",
    ) as profiler:
        training_step(model, batch, generator)
    synchronize(device)
    rows = []
    for event in profiler.key_averages():
        device_time = float(getattr(event, "device_time_total", 0.0))
        cpu_time = float(getattr(event, "cpu_time_total", 0.0))
        if device_time <= 0.0 and cpu_time <= 0.0:
            continue
        rows.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "device_time_us": device_time,
                "cpu_time_us": cpu_time,
                "self_device_time_us": float(
                    getattr(event, "self_device_time_total", 0.0)
                ),
                "self_cpu_time_us": float(
                    getattr(event, "self_cpu_time_total", 0.0)
                ),
            }
        )
    rows.sort(
        key=lambda row: row["self_device_time_us"]
        + row["self_cpu_time_us"],
        reverse=True,
    )
    return rows[:40]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(19)
    store = PackedCaseStore(args.cache)
    indices = store.indices(args.block_count)[: args.batch_size]
    batch = store.batch(indices, device=device)
    model = build_chip_model(production_config()).to(device)
    model.train()
    generator = torch.Generator(device=device).manual_seed(19)
    with torch.no_grad():
        context_ms = timed(
            lambda: model.prepare_context(batch),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    training_ms = timed(
        lambda: training_step(model, batch, generator),
        device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    model.eval()
    def run_sample(compiled_projector: bool) -> tuple[torch.Tensor, dict]:
        sample_generator = torch.Generator(device=device).manual_seed(19)
        return sample_hierarchy_flow(
            model,
            batch,
            sampling_steps=args.sample_steps,
            generator=sample_generator,
            projector=True,
            projector_iterations=8,
            compile_projector=compiled_projector,
        )

    with torch.no_grad():
        eager_sample, _ = run_sample(False)
        compiled_sample = None
        compiled_max_difference = None
        if args.compile_projector:
            compiled_sample, _ = run_sample(True)
            compiled_max_difference = float(
                (eager_sample - compiled_sample).abs().max()
            )
    sampling_ms = timed(
        lambda: run_sample(False),
        device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    compiled_sampling_ms = None
    if args.compile_projector:
        compiled_sampling_ms = timed(
            lambda: run_sample(True),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    latent = batch["target_latent"].clone()
    projection_ms = timed(
        lambda: project_hard_constraints(
            latent,
            batch["target_latent"],
            batch["constraints"],
            batch["areas"],
            iterations=8,
        ),
        device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    kernels = kernel_profile(model, batch, generator, device)
    report = {
        "version": "chip-v46-typed-operator-profile-v1",
        "status": "complete",
        "device": str(device),
        "block_count": args.block_count,
        "batch_size": len(indices),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timings_ms": {
            "typed_context": context_ms,
            "training_step": training_ms,
            "sampling_nfe": sampling_ms,
            "sampling_nfe_compiled_projector": compiled_sampling_ms,
            "sampling_compiled_max_abs_difference": compiled_max_difference,
            "projector_8": projection_ms,
        },
        "cuda_memory_bytes": (
            {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
            if device.type == "cuda"
            else {}
        ),
        "top_operators": kernels,
        "architecture": production_config(),
        "scope": (
            "Profile only. No quality claim and no operator replacement is "
            "implied by this artifact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
