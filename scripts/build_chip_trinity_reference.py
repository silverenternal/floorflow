#!/usr/bin/env python3
"""Build a checkpoint whose block flow is the pinned Trinity backbone identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.hierarchical_flow import HierarchicalTokenFlowModel
from floorflow.chip_flow.training_runtime import save_torch_atomic
from floorflow.chip_flow.trinity_backbone_init import (
    initialize_trinity_backbone,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=ROOT / "third_party/Trinity/checkpoints/d192",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    model = HierarchicalTokenFlowModel(
        hidden=192,
        heads=3,
        depth=12,
        typed_layers=1,
        head_dim=64,
        use_hierarchy=False,
        use_netlist=True,
        projector=True,
        projector_iterations=4,
    )
    provenance = initialize_trinity_backbone(model, args.release)
    config = {
        "architecture": "typed_hierarchy",
        "hidden": 192,
        "heads": 3,
        "depth": 12,
        "head_dim": 64,
        "graph_pe_k": 2,
        "typed_layers": 1,
        "typed_mode": "typed",
        "use_hierarchy": False,
        "use_netlist": True,
        "projector": True,
        "projector_iterations": 4,
        "projector_boundary_axes": False,
        "projector_boundary_lock_corners": True,
        "hierarchy_source_levels": "random",
    }
    state = model.state_dict()
    save_torch_atomic(
        args.output,
        {
            "step": 0,
            "model": state,
            "ema": state,
            "config": config,
            "initialization_provenance": provenance,
            "reference_contract": {
                "block_flow": "pinned Trinity d192 backbone",
                "typed_hierarchy_residual": "disabled with use_hierarchy=false",
                "projection_consistent": False,
            },
        },
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
