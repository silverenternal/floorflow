#!/usr/bin/env python3
"""Measure isolated typed-channel and cross-attention training scopes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from argparse import Namespace
from pathlib import Path

import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.train_objective import supervised_loss
from floorflow.chip_flow.training_scopes import (
    apply_training_scope,
    parse_training_scope,
)


def _batch(batch: int, count: int, pins: int) -> dict[str, torch.Tensor]:
    constraints = torch.zeros(batch, count, 5)
    constraints[:, : max(1, count // 8), 0] = 1.0
    constraints[:, : max(1, count // 4), 2] = 1.0
    constraints[:, :, 3] = torch.arange(count) % 4 + 1
    constraints[:, :, 4] = (torch.arange(count) % 15 + 1).float()
    pairs = max(count, count * 3)
    source = torch.arange(pairs) % count
    target = (source + 1 + torch.arange(pairs) // count) % count
    pin_block = torch.arange(pins) % count
    return {
        "areas": torch.rand(batch, count) + 0.25,
        "boxes": torch.rand(batch, count, 4) * 0.2 + 0.05,
        "source_latent": torch.randn(batch, count, 3),
        "target_latent": torch.randn(batch, count, 3),
        "constraints": constraints,
        "features": torch.randn(batch, count, 19),
        "b2b": torch.stack(
            [source.float(), target.float(), torch.rand(pairs) + 0.1],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "p2b": torch.stack(
            [
                torch.arange(pins).float(),
                pin_block.float(),
                torch.rand(pins) + 0.1,
            ],
            dim=-1,
        )[None].expand(batch, -1, -1),
        "pins": torch.rand(batch, pins, 2),
        "n_b2b": torch.full((batch,), pairs),
        "n_p2b": torch.full((batch,), pins),
        "n_pins": torch.full((batch,), pins),
        "global_spec": torch.randn(batch, 7),
        "global_spec_mask": torch.ones(batch, dtype=torch.bool),
        "node_features": torch.randn(batch, count, 4),
        "node_feature_mask": torch.ones(batch, count, dtype=torch.bool),
        "port_features": torch.randn(batch, pins, 4),
        "port_feature_mask": torch.ones(batch, pins, dtype=torch.bool),
    }


def _objective_args() -> Namespace:
    return Namespace(
        time_std=1.0,
        flow_huber_beta=0.1,
        path_steps=0,
        path_weight=0.2,
        physics_weight=1.0,
        overlap_weight_scale=10.0,
        decode_weight=1.0,
        rho_weight=0.0,
        clean_weight=0.1,
        sparsity_weight=0.01,
        route_balance_weight=0.01,
        hierarchy_global_weight=0.1,
        hierarchy_structural_weight=0.1,
        hierarchy_port_weight=0.1,
        hierarchy_source_mode="random",
        hierarchy_source_noise=0.0,
        hierarchy_source_levels="random",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("typed", "full"), default="typed")
    parser.add_argument("--scope", action="append", dest="scopes")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not args.scopes:
        args.scopes = ["all", "channels", "fusion"]
    device = torch.device(args.device)
    torch.manual_seed(31)
    model = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=2,
        head_dim=16,
        typed_layers=2,
        relation_specific=True,
        degree_expert_conditioning=True,
        global_spec_conditioning=True,
        node_semantic_dim=4,
        port_semantic_dim=4,
    ).to(device)
    batch = {
        name: value.to(device)
        for name, value in _batch(batch=8, count=48, pins=64).items()
    }
    rows = []
    for scope_name in args.scopes:
        scope = parse_training_scope(scope_name)
        stats = apply_training_scope(model, scope)
        durations = []
        for index in range(args.repeats + 1):
            torch.manual_seed(100 + index)
            model.zero_grad(set_to_none=True)
            _synchronize(device)
            started = time.perf_counter()
            if args.mode == "typed":
                context = model.prepare_context(batch)
                loss = context.typed.block_tokens.float().square().mean()
            else:
                loss, _ = supervised_loss(
                    model,
                    batch,
                    _objective_args(),
                    device,
                )
            loss.backward()
            _synchronize(device)
            if index:
                durations.append(time.perf_counter() - started)
        rows.append(
            {
                "scope": scope.name,
                "trainable_parameters": stats["trainable_parameters"],
                "median_forward_backward_seconds": statistics.median(durations),
                "samples": durations,
            }
        )
    result = {
        "device": str(device),
        "mode": args.mode,
        "batch": 8,
        "blocks": 48,
        "pins": 64,
        "repeats": args.repeats,
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
