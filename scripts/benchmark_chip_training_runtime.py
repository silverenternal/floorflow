#!/usr/bin/env python3
"""Benchmark atomic chip-training optimizations without changing the objective."""

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
from floorflow.chip_flow.projections import project_conditioned_latent
from floorflow.chip_flow.training_runtime import (
    initial_ema_state,
    make_optimizer,
    update_ema_state_,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_training_framework_benchmark_v1.json",
    )
    return parser.parse_args()


def reference_projection(
    latent: torch.Tensor,
    target: torch.Tensor,
    constraints: torch.Tensor,
) -> torch.Tensor:
    """Retain the original loop as a benchmark and equivalence reference."""

    projected = latent.clone()
    fixed = constraints[..., 0] != 0
    preplaced = constraints[..., 1] != 0
    anchored = fixed | preplaced
    projected[..., 2] = torch.where(
        anchored,
        target[..., 2],
        projected[..., 2],
    )
    projected[..., :2] = torch.where(
        preplaced[..., None],
        target[..., :2],
        projected[..., :2],
    )
    mib_ids = constraints[..., 2]
    for batch_index in range(latent.shape[0]):
        for group in torch.unique(mib_ids[batch_index][mib_ids[batch_index] > 0]):
            members = mib_ids[batch_index] == group
            anchors = members & anchored[batch_index]
            rho = (
                target[batch_index, anchors, 2].median()
                if anchors.any()
                else projected[batch_index, members, 2].median()
            )
            projected[batch_index, members, 2] = rho
    return projected


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


def projection_benchmark(
    batch: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    target = batch["target_latent"]
    constraints = batch["constraints"]
    base = target.clone()
    report = {}
    for name, fn in (
        ("reference", reference_projection),
        ("vectorized", project_conditioned_latent),
    ):
        def step(call=fn) -> None:
            latent = base.clone().requires_grad_(True)
            call(latent, target, constraints).square().mean().backward()

        report[f"{name}_ms"] = timed(
            step,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )

    left = base.clone().requires_grad_(True)
    right = base.clone().requires_grad_(True)
    reference_projection(left, target, constraints).square().mean().backward()
    project_conditioned_latent(right, target, constraints).square().mean().backward()
    report["value_max_abs_difference"] = float(
        (
            reference_projection(base, target, constraints)
            - project_conditioned_latent(base, target, constraints)
        )
        .abs()
        .max()
    )
    report["gradient_max_abs_difference"] = float(
        (left.grad - right.grad).abs().max()
    )
    report["speedup"] = report["reference_ms"] / report["vectorized_ms"]
    return report


def optimizer_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    report = {}
    for fused in (False, True):
        if fused and device.type != "cuda":
            continue
        optimizer = make_optimizer(
            model,
            lr=2.0e-4,
            weight_decay=1.0e-3,
            fused=fused,
        )
        optimizer.zero_grad(set_to_none=False)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        report[f"fused_{str(fused).lower()}_ms"] = timed(
            optimizer.step,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    if "fused_true_ms" in report:
        report["speedup"] = report["fused_false_ms"] / report["fused_true_ms"]
    return report


def ema_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    model_state = model.state_dict()
    ema = initial_ema_state(model)
    reference = {name: value.detach().clone() for name, value in ema.items()}
    for value in model_state.values():
        value.add_(0.01)

    def reference_update() -> None:
        for name, value in reference.items():
            if torch.is_floating_point(value):
                value.lerp_(model_state[name].detach(), 1.0 - 0.9995)
            else:
                value.copy_(model_state[name])

    report = {
        "reference_ms": timed(
            reference_update,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "foreach_ms": timed(
            lambda: update_ema_state_(ema, model_state, 0.9995),
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    report["speedup"] = report["reference_ms"] / report["foreach_ms"]
    report["max_abs_difference"] = max(
        float((ema[name] - reference[name]).abs().max())
        for name in ema
        if torch.is_floating_point(ema[name])
    )
    return report


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    store = PackedCaseStore(
        ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train"
    )
    indices = store.indices(args.block_count)[: args.batch_size]
    batch = store.batch(indices, device=device)
    gather_ms = timed(
        lambda: store.batch(indices, device=device),
        device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    model = build_chip_model(
        {
            "architecture": "graph_slot",
            "feature_dim": 19,
            "hidden": 192,
            "global_dim": 192,
            "target_layers": 4,
            "condition_layers": 2,
            "decoder_layers": 4,
            "heads": 6,
            "graph_conditioning": True,
        }
    ).to(device)
    result = {
        "status": "complete",
        "scope": "atomic runtime benchmark; no quality claim",
        "device": str(device),
        "block_count": args.block_count,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "data_gather_ms": gather_ms,
        "projection": projection_benchmark(batch, device, args),
        "optimizer": optimizer_benchmark(model, device, args),
        "ema": ema_benchmark(model, device, args),
        "framework_availability": {
            "pytorch_lightning": _installed("pytorch_lightning"),
            "lightning": _installed("lightning"),
            "accelerate": _installed("accelerate"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _installed(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
