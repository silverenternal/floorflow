#!/usr/bin/env python3
"""CPU causality probe for NFE1 hierarchy conditioning and commit semantics."""

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
from floorflow.chip_flow.training_runtime import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_nfe1_hierarchy_causality_v1.json",
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
        "p2b": torch.tensor([[[0.0, 0.0, 1.0]]] * 2),
        "pins": torch.rand(2, 1, 2),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([1, 1]),
        "n_pins": torch.tensor([1, 1]),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(19)
    model = HierarchicalTokenFlowModel(
        hidden=32,
        heads=4,
        depth=1,
        head_dim=16,
        typed_layers=1,
        structured_heads=True,
        complexity_conditioning=True,
        structural_relations=True,
        relation_specific=True,
        coarse_commit_fraction=0.3,
        projector_grid_size=8,
    )
    suite = model.structured_heads
    assert suite is not None and suite.hierarchy_coupling is not None
    suite.hierarchy_coupling.global_gate.data.fill_(1.0)
    suite.hierarchy_coupling.structure_gate.data.fill_(1.0)
    torch.nn.init.normal_(
        suite.hierarchy_coupling.global_reduce[-1].weight,
        std=0.02,
    )
    torch.nn.init.normal_(
        suite.hierarchy_coupling.structure_reduce[-1].weight,
        std=0.02,
    )
    batch = _batch()
    context = model.prepare_context(batch)
    target = build_hierarchy_targets(batch, context)
    time = torch.full((2,), 0.5)
    base = model(target, time, context)
    coupler = suite.hierarchy_coupling
    base_residual, _ = coupler(
        context.typed.block_tokens,
        target.block_state,
        context.coarse_membership,
        context.typed.structural_mask,
        target.global_state,
        target.structural_state,
        context.typed.structural_tokens,
        time,
    )
    global_residual, _ = coupler(
        context.typed.block_tokens,
        target.block_state,
        context.coarse_membership,
        context.typed.structural_mask,
        target.global_state + 2.0,
        target.structural_state,
        context.typed.structural_tokens,
        time,
    )
    structure_residual, _ = coupler(
        context.typed.block_tokens,
        target.block_state,
        context.coarse_membership,
        context.typed.structural_mask,
        target.global_state,
        target.structural_state + 2.0,
        context.typed.structural_tokens,
        time,
    )
    report = {
        "version": "chip-nfe1-hierarchy-causality-v1",
        "status": "pass_cpu_contract_only",
        "coarse_commit": {
            "nfe1_fraction_0p3_threshold": coarse_commit_threshold(0.3, 1),
            "nfe4_fraction_0p3_threshold": coarse_commit_threshold(0.3, 4),
        },
        "same_call_sensitivity": {
            "global_block_max_abs": float(
                (global_residual - base_residual).abs().max().detach()
            ),
            "structural_block_max_abs": float(
                (structure_residual - base_residual).abs().max().detach()
            ),
        },
        "complexity_token_enabled": model.typed.complexity is not None,
        "structural_relation_shape": list(
            base.structural_relations.shape
            if base.structural_relations is not None
            else ()
        ),
        "boundary": "Random-initialized CPU causality only; no quality claim.",
    }
    assert report["coarse_commit"]["nfe1_fraction_0p3_threshold"] == 1
    assert report["same_call_sensitivity"]["global_block_max_abs"] > 0.0
    assert report["same_call_sensitivity"]["structural_block_max_abs"] > 0.0
    write_json_atomic(args.output, json.loads(json.dumps(report, default=str)))
    print(args.output)


if __name__ == "__main__":
    main()
