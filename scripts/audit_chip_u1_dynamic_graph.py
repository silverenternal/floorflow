#!/usr/bin/env python3
"""Audit the opt-in U1 state-to-edge relay before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.sparse_layers import prepare_state_edge_cache

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_u1_dynamic_graph_representation_probe_v1.json",
    )
    return parser.parse_args()


def build_model(dynamic: bool) -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=192,
        heads=3,
        depth=10,
        typed_layers=1,
        head_dim=64,
        projector=True,
        projector_iterations=4,
        dynamic_graph_relay=dynamic,
    ).eval()


def remap_edges(batch: dict, permutation: torch.Tensor) -> dict:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    changed = {key: value for key, value in batch.items()}
    changed["areas"] = batch["areas"][:, permutation]
    changed["boxes"] = batch["boxes"][:, permutation]
    changed["constraints"] = batch["constraints"][:, permutation]
    changed["features"] = batch["features"][:, permutation]
    changed["target_latent"] = batch["target_latent"][:, permutation]
    changed["b2b"] = batch["b2b"].clone()
    changed["b2b"][..., :2] = inverse[
        batch["b2b"][..., :2].long()
    ].to(batch["b2b"].dtype)
    changed["p2b"] = batch["p2b"].clone()
    changed["p2b"][..., 1] = inverse[
        batch["p2b"][..., 1].long()
    ].to(batch["p2b"].dtype)
    return changed


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261030)
    store = PackedCaseStore(args.cache)
    block_count = min(store.by_block_count)
    indices = store.indices(block_count)[:4]
    batch = store.batch(indices, device="cpu")
    batch["_indices"] = indices
    base = build_model(False)
    dynamic = build_model(True)
    dynamic.load_state_dict(base.state_dict(), strict=False)
    base_context = base.prepare_context(batch)
    dynamic_context = dynamic.prepare_context(batch)
    targets = build_hierarchy_targets(batch, dynamic_context)
    state = targets.block_state
    time = torch.full((batch["areas"].shape[0],), 0.5)
    base_prediction = base(targets, time, base_context).block_state
    dynamic_prediction = dynamic(targets, time, dynamic_context).block_state
    assert dynamic.dynamic_relay is not None
    assert dynamic_context.edge_state is not None
    dynamic.dynamic_relay.gate.data.fill_(1.0)
    torch.nn.init.normal_(dynamic.dynamic_relay.output.weight, std=0.02)

    def relay(tokens: torch.Tensor, edge_batch: dict) -> torch.Tensor:
        cache = prepare_state_edge_cache(
            areas=edge_batch["areas"],
            b2b=edge_batch["b2b"],
            p2b=edge_batch["p2b"],
            pins=edge_batch["pins"],
            n_b2b=edge_batch["n_b2b"],
            n_p2b=edge_batch["n_p2b"],
            n_pins=edge_batch["n_pins"],
        )
        return dynamic.dynamic_relay(
            tokens,
            cache=cache,
        )

    original = relay(state, batch)
    shifted = relay(state + 0.25, batch)
    permutation = torch.randperm(state.shape[1])
    permuted_batch = remap_edges(batch, permutation)
    permuted = relay(state[:, permutation], permuted_batch)
    checks = {
        "zero_init_equivalence": torch.equal(base_prediction, dynamic_prediction),
        "state_sensitivity": (
            float((shifted - original).abs().mean().detach()) > 1.0e-6
        ),
        "permutation_equivariance": torch.allclose(
            original[:, permutation],
            permuted,
            atol=2.0e-5,
            rtol=2.0e-5,
        ),
        "sparse_edges_present": (
            int(batch["n_b2b"].sum()) > 0 and int(batch["n_p2b"].sum()) > 0
        ),
    }
    report = {
        "version": "chip-u1-dynamic-graph-representation-probe-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "parameters": {
            "base": sum(value.numel() for value in base.parameters()),
            "dynamic": sum(value.numel() for value in dynamic.parameters()),
        },
        "edges": {
            "b2b": int(batch["n_b2b"].sum()),
            "p2b": int(batch["n_p2b"].sum()),
            "pins": int(batch["n_pins"].sum()),
        },
        "sensitivity_mean_abs": float(
            (shifted - original).abs().mean().detach()
        ),
        "permutation_max_abs": float(
            (original[:, permutation] - permuted).abs().max().detach()
        ),
        "scope": "CPU representation probe only; no quality claim.",
    }
    report["parameters"]["added"] = (
        report["parameters"]["dynamic"] - report["parameters"]["base"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
