#!/usr/bin/env python3
"""CPU audit of coupled L0-L3 hierarchy on a real FloorSet batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--block-count", type=int, default=31)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_hierarchical_flow_cpu_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    store = PackedCaseStore(args.cache)
    batch = store.batch(
        store.indices(args.block_count)[: args.cases],
        device="cpu",
    )
    model = HierarchicalTokenFlowModel()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time_value = torch.full((args.cases,), 0.5)
    started = time.perf_counter()
    prediction = model(target, time_value, context)
    base = model.backbone(
        target.block_state,
        time_value,
        context.denoiser_cond,
    )
    zero_init_error = float(
        (prediction.block_state.detach() - base.detach()).abs().max()
    )
    torch.nn.init.normal_(model.hierarchy_to_latent.weight, std=0.02)
    torch.nn.init.normal_(model.backbone.final.linear.weight, std=0.02)
    model.typed.cross_attention.gate.data.fill_(1.0)
    coupled = model(target, time_value, context)
    coupled_delta = float(
        (coupled.block_state.detach() - base.detach()).abs().mean()
    )
    (
        coupled.block_state.square().mean()
        + coupled.global_state.square().mean()
    ).backward()
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "zero_init_block_equivalence": zero_init_error == 0.0,
        "nonzero_coupled_residual": coupled_delta > 0.0,
        "finite_forward": bool(torch.isfinite(coupled.block_state).all()),
        "typed_and_hierarchy_gradients": bool(
            model.typed.port_input.weight.grad is not None
            and model.global_head[-1].weight.grad is not None
            and model.hierarchy_to_latent.weight.grad is not None
        ),
        "parameter_budget_below_9m": parameters < 9_000_000,
    }
    report = {
        "status": "pass" if all(checks.values()) else "redesign",
        "scope": "CPU L0-L3 hierarchy audit on a real FloorSet batch",
        "block_count": args.block_count,
        "cases": args.cases,
        "parameters": parameters,
        "forward_backward_seconds": elapsed,
        "zero_init_block_error": zero_init_error,
        "coupled_block_delta": coupled_delta,
        "state_contract": {
            "L0": "global bbox center and width/height",
            "L1": "cluster(4), MIB(1), boundary(4) mean latent states",
            "L2": "per-block latent trajectory",
            "L3": "pin-relative connected-block alignment; pins remain read-only"
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
