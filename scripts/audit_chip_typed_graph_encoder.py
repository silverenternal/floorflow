#!/usr/bin/env python3
"""CPU audit of the typed netlist graph encoder on a real FloorSet batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.graph_conditioning import graph_conditioned_features
from floorflow.chip_flow.typed_graph_encoder import TypedGraphEncoder

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/train",
    )
    parser.add_argument("--block-count", type=int, default=31)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_typed_graph_cpu_v1.json",
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
    features, _, _ = graph_conditioned_features(batch)
    model = TypedGraphEncoder()
    started = time.perf_counter()
    output = model(
        features,
        batch["constraints"],
        areas=batch["areas"],
        b2b=batch["b2b"],
        p2b=batch["p2b"],
        pins=batch["pins"],
        n_b2b=batch["n_b2b"],
        n_p2b=batch["n_p2b"],
        n_pins=batch["n_pins"],
    )
    output.block_tokens.square().mean().backward()
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "finite_forward_and_gradients": bool(
            torch.isfinite(output.block_tokens).all()
            and model.block_input[0].weight.grad is not None
            and torch.isfinite(model.block_input[0].weight.grad).all()
        ),
        "typed_mask_is_sparse": (
            0.0 < float(output.diagnostics["typed_edge_fraction"]) < 1.0
        ),
        "parameter_budget_below_2m": parameters < 2_000_000,
        "no_unavailable_channels": True,
    }
    report = {
        "status": "pass" if all(checks.values()) else "redesign",
        "scope": "CPU typed netlist graph audit on a real FloorSet batch",
        "block_count": args.block_count,
        "cases": args.cases,
        "parameters": parameters,
        "forward_backward_seconds": elapsed,
        "diagnostics": {
            name: float(value)
            for name, value in output.diagnostics.items()
        },
        "channel_contract": {
            "available": [
                "global boundary context",
                "b2b/p2b netlist",
                "cluster",
                "MIB",
                "boundary code",
                "fixed/preplaced constraints",
                "pins/terminals"
            ],
            "unavailable": [
                "opening",
                "port direction",
                "halo geometry",
                "routing congestion"
            ]
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
