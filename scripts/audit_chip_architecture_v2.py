#!/usr/bin/env python3
"""CPU contract audit for all completed chip architecture upgrades."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from floorflow.chip_flow.hierarchical_advance import coarse_commit_threshold
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.sparse_layers import prepare_state_edge_cache
from floorflow.chip_flow.spatial_occupancy_coupler import SpatialOccupancyCoupler
from floorflow.chip_flow.structural_relations import structural_relation_features
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_architecture_v2_contract_audit.json",
    )
    return parser.parse_args()


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    constraints = torch.zeros(2, count, 5)
    constraints[..., 3] = torch.tensor([1, 1, 2, 2, 3, 3])
    constraints[:, :2, 4] = 1
    return {
        "areas": boxes[..., 2] * boxes[..., 3],
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 19),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor([[[0.0, 1.0, 1.0], [1.0, 2.0, 0.5]]] * 2),
        "p2b": torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 0.5]]] * 2),
        "pins": torch.rand(2, 2, 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def _base_model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        projector_grid_size=8,
    )


def _v2_model() -> HierarchicalTokenFlowModel:
    return HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        relation_specific=True,
        structural_relations=True,
        complexity_conditioning=True,
        degree_expert_conditioning=True,
        coarse_commit_fraction=0.3,
        structured_heads=True,
        structured_legalizer_apply=True,
        structured_legalizer_strength=0.5,
        dynamic_graph_input=True,
        spatial_route_field=True,
        projector_grid_size=8,
    )


def _route_sensitivity() -> bool:
    torch.manual_seed(7)
    areas = torch.ones(1, 4)
    state = torch.randn(1, 4, 3)
    tokens = torch.randn(1, 4, 16)
    cache = prepare_state_edge_cache(
        areas=areas,
        b2b=torch.tensor([[[0.0, 1.0, 1.0]]]),
        p2b=torch.tensor([[[0.0, 0.0, 1.0]]]),
        pins=torch.tensor([[[0.1, 0.2]]]),
        n_b2b=torch.tensor([1]),
        n_p2b=torch.tensor([1]),
        n_pins=torch.tensor([1]),
    )
    module = SpatialOccupancyCoupler(16, route_aware=True)
    module.gate.data.fill_(1.0)
    first, _ = module(tokens, state, areas, cache=cache)
    second, _ = module(
        tokens,
        state,
        areas,
        cache={**cache, "pins": torch.tensor([[0.9, 0.8]])},
    )
    return not torch.equal(first, second)


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    base = _base_model()
    upgraded = _v2_model()
    incompatible = upgraded.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    upgraded_context = upgraded.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)
    expected = base(target, time, base_context)
    actual = upgraded(target, time, upgraded_context)
    relation_target, relation_mask = structural_relation_features(
        batch["boxes"],
        batch["areas"],
        batch["constraints"],
    )
    report = {
        "version": "chip-architecture-v2-contract-audit",
        "status": "pass_cpu_contract_only",
        "zero_gate_equivalence": {
            "block_max_abs": float(
                (actual.block_state - expected.block_state).abs().max().detach()
            ),
            "global_max_abs": float(
                (actual.global_state - expected.global_state).abs().max().detach()
            ),
            "structural_max_abs": float(
                (actual.structural_state - expected.structural_state)
                .abs()
                .max()
                .detach()
            ),
            "port_max_abs": float(
                (actual.port_state - expected.port_state).abs().max().detach()
            ),
        },
        "missing_parameters_are_zero_gated": sorted(incompatible.missing_keys),
        "relation_target": {
            "shape": list(relation_target.shape),
            "active_slots": int(relation_mask.sum()),
            "finite": bool(torch.isfinite(relation_target).all()),
        },
        "route_pin_sensitivity": _route_sensitivity(),
        "nfe1_coarse_commit_step": coarse_commit_threshold(0.3, 1),
        "complexity_conditioning": upgraded.typed.complexity is not None,
        "legalizer_response_action": upgraded.structured_legalizer_apply,
        "legalizer_response_trust_radius": (
            upgraded.structured_legalizer_max_displacement
        ),
        "ports": {
            "structural_relations": actual.structural_relations is not None,
            "legalizer_response": actual.legalizer_displacement is not None,
        },
        "parameters": {
            "base": sum(parameter.numel() for parameter in base.parameters()),
            "upgraded": sum(parameter.numel() for parameter in upgraded.parameters()),
        },
        "boundary": "No training, quality or GPU inference claim.",
    }
    assert all(
        value == 0.0 for value in report["zero_gate_equivalence"].values()
    )
    assert report["route_pin_sensitivity"]
    write_json_atomic(args.output, json.loads(json.dumps(report, default=str)))
    print(args.output)


if __name__ == "__main__":
    main()
