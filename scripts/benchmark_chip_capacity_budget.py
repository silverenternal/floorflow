#!/usr/bin/env python3
"""Measure production typed-hierarchy capacity and inference budgets."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.flow_adapter import build_chip_model, sample_for_batch
from floorflow.chip_flow.hierarchical_flow import build_hierarchy_targets
from floorflow.chip_flow.training_runtime import write_json_atomic

MODEL_KEYS = (
    "hidden",
    "heads",
    "depth",
    "head_dim",
    "graph_pe_k",
    "typed_layers",
    "typed_mode",
    "use_hierarchy",
    "use_netlist",
    "netlist_edge_gate",
    "projector",
    "projector_iterations",
    "projector_boundary_axes",
    "projector_boundary_lock_corners",
    "hierarchy_global_weight",
    "hierarchy_structural_weight",
    "hierarchy_port_weight",
    "hierarchy_bbox_weight",
    "hierarchy_bbox_decay_steps",
    "outline_conditioning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
        "typed_hierarchy_bbox_anneal_seed19_20k/checkpoints/final.pt",
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--block-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--widths", type=int, nargs="+", default=[192, 256, 384])
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_capacity_budget_v1.json",
    )
    return parser.parse_args()


def synced_time(fn, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(sorted(samples)[len(samples) // 2])


def model_config(base: dict, width: int) -> dict:
    config = {
        key: copy.deepcopy(base[key])
        for key in MODEL_KEYS
        if key in base
    }
    config.update(
        {
            "architecture": "typed_hierarchy",
            "hidden": int(width),
            "heads": max(1, int(width) // 64),
            "depth": int(base.get("depth", 10)),
            "head_dim": int(base.get("head_dim", 64)),
            "typed_layers": int(base.get("typed_layers", 1)),
        }
    )
    return config


def model_flops(model, batch, target, device: torch.device) -> int:
    context = model.prepare_context(batch)
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as counter:
        prediction = model(
            target,
            torch.full((batch["areas"].shape[0],), 0.5, device=device),
            context,
        )
        loss = sum(value.float().sum() for value in prediction.__dict__.values() if torch.is_tensor(value))
        loss.backward()
    return int(counter.get_total_flops())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache = args.cache or ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train"
    store = PackedCaseStore(cache)
    indices = store.indices(args.block_count)[: args.batch_size]
    batch = store.batch(indices, device=device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_config = checkpoint["config"]
    rows = []
    for width in args.widths:
        config = model_config(base_config, width)
        model = build_chip_model(config).to(device)
        model.train()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        context = model.prepare_context(batch)
        target = build_hierarchy_targets(batch, context)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_flops = model_flops(model, batch, target, device)
        training_ms = synced_time(
            lambda model=model, target=target: model_flops(
                model,
                batch,
                target,
                device,
            ),
            device,
            args.warmup,
            args.iterations,
        )
        generator = torch.Generator(device=device).manual_seed(19)
        sample_flops_start = 0
        with FlopCounterMode(display=False) as counter:
            sample_for_batch(
                model,
                batch,
                sampling_steps=args.nfe,
                generator=generator,
            )
        sample_flops_start = int(counter.get_total_flops())
        sample_ms = synced_time(
            lambda model=model: sample_for_batch(
                model,
                batch,
                sampling_steps=args.nfe,
                generator=torch.Generator(device=device).manual_seed(19),
            ),
            device,
            args.warmup,
            args.iterations,
        )
        peak = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        )
        rows.append(
            {
                "width": width,
                "heads": config["heads"],
                "parameters": parameters,
                "training_forward_backward_flops": training_flops,
                "training_forward_backward_ms": training_ms,
                "sample_nfe_flops": sample_flops_start,
                "sample_nfe_ms": sample_ms,
                "peak_gpu_memory_mb": peak,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "version": "chip-capacity-budget-v1",
        "status": "preflight_no_quality_claim",
        "checkpoint": str(args.checkpoint),
        "cache": str(cache),
        "block_count": args.block_count,
        "batch_size": args.batch_size,
        "nfe": args.nfe,
        "device": str(device),
        "rows": rows,
        "claim_boundary": "This measures budget and runtime only; it does not compare placement quality.",
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
