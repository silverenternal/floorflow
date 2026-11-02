#!/usr/bin/env python3
"""Small cuda:0 readiness and profile probe for the v2 chip architecture."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import sample_for_batch
from floorflow.chip_flow.hierarchical_flow import build_hierarchy_targets
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/m27_training_seed_replication_v1"
            / "m27_seed21_5k/checkpoints/final.pt"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--block-count", type=int, default=47)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v2_gpu_readiness_v1.json",
    )
    return parser.parse_args()


def _v2_config(config: dict) -> dict:
    result = dict(config)
    result.update(
        {
            "relation_specific": True,
            "structural_relations": True,
            "complexity_conditioning": True,
            "degree_expert_conditioning": True,
            "coarse_commit_fraction": 0.3,
            "dynamic_graph_input": True,
            "spatial_route_field": True,
            "structured_heads": True,
            "structured_hierarchy_coupling": True,
            "structured_factorized_block": True,
            "structured_dual_ports": True,
            "structured_legalizer_response": True,
            "structured_legalizer_apply": True,
            "structured_legalizer_strength": 0.5,
            "structured_legalizer_max_displacement": 0.1,
        }
    )
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(function, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1.0e3)
    return float(statistics.median(samples))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this readiness probe")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = load_chip_checkpoint(checkpoint, device, use_ema=True)
    v2_checkpoint = {
        **checkpoint,
        "config": _v2_config(checkpoint["config"]),
    }
    v2 = load_chip_checkpoint(v2_checkpoint, device, use_ema=True)
    store = PackedCaseStore(args.cache)
    indices = store.indices(args.block_count)[: args.batch_size]
    batch = store.batch(indices, device=device)
    batch["_indices"] = torch.as_tensor(indices, device=device)
    base_context = base.prepare_context(batch)
    context = v2.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    times = torch.full((len(indices),), 0.5, device=device)
    with torch.no_grad():
        base_prediction = base(target, times, base_context)
        v2_prediction = v2(target, times, context, apply_legalizer_action=True)
    equivalence = {
        "block_max_abs": float(
            (v2_prediction.block_state - base_prediction.block_state)
            .abs()
            .max()
            .detach()
        ),
        "global_max_abs": float(
            (v2_prediction.global_state - base_prediction.global_state)
            .abs()
            .max()
            .detach()
        ),
    }

    def train_step() -> None:
        step_context = v2.prepare_context(batch)
        step_target = build_hierarchy_targets(batch, step_context)
        prediction = v2(
            step_target,
            times,
            step_context,
            apply_legalizer_action=True,
        )
        loss = (
            prediction.block_state.float().square().mean()
            + prediction.structural_relations.float().square().mean()
            + prediction.legalizer_displacement.float().square().mean()
        )
        loss.backward()
        v2.zero_grad(set_to_none=True)

    context_ms = timed(
        lambda: v2.prepare_context(batch),
        device,
        args.warmup,
        args.iterations,
    )
    train_step_ms = timed(
        train_step,
        device,
        args.warmup,
        args.iterations,
    )
    with torch.no_grad():
        sample_ms = timed(
            lambda: sample_for_batch(
                v2,
                batch,
                sampling_steps=1,
                generator=torch.Generator(device=device).manual_seed(19),
            ),
            device,
            args.warmup,
            args.iterations,
        )
    torch.cuda.synchronize(device)
    report = {
        "version": "chip-v2-gpu-readiness-v1",
        "status": (
            "pass_numerically"
            if max(equivalence.values()) <= 1.0e-5
            else "failed_equivalence"
        ),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "block_count": args.block_count,
        "batch_size": len(indices),
        "zero_gate_equivalence": equivalence,
        "equivalence_tolerance": 1.0e-5,
        "context_ms": context_ms,
        "forward_backward_ms": train_step_ms,
        "nfe1_sample_ms": sample_ms,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "v2_modules": {
            "typed_complexity": v2.typed.complexity is not None,
            "degree_expert": v2.typed.degree_expert is not None,
            "relation_specific": v2.typed.relation_specific,
            "structural_relations": v2.relation_head is not None,
            "dynamic_graph_input": v2.dynamic_graph_input,
            "spatial_route_field": v2.spatial_route_field,
            "structured_heads": v2.structured_heads is not None,
            "legalizer_action": v2.structured_legalizer_apply,
        },
        "boundary": "Short readiness/profile probe only; no quality claim.",
    }
    write_json_atomic(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
