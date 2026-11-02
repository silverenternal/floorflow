#!/usr/bin/env python3
"""CPU smoke for the reusable graph-conditioned global-slot path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.graph_conditioning import graph_conditioned_features
from floorflow.chip_flow.slot_model import ChipSlotFlowModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "outputs/cache/chip_official_test_100_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_graph_conditioning_cpu_v1.json",
    )
    parser.add_argument("--blocks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    store = PackedCaseStore(args.cache)
    indices = store.indices(args.blocks)[:2]
    batch = store.batch(indices, device="cpu")
    features, adjacency, key_mask = graph_conditioned_features(batch)
    model = ChipSlotFlowModel(
        feature_dim=19,
        hidden=32,
        global_dim=32,
        target_layers=1,
        condition_layers=1,
        decoder_layers=1,
        heads=4,
        graph_conditioning=True,
    )
    with_graph = model.encode_target(
        batch["target_latent"],
        features,
        adjacency,
        key_mask,
    )
    no_graph = model.encode_target(
        batch["target_latent"],
        features,
    )
    permutation = torch.arange(args.blocks - 1, -1, -1)
    permuted_features = features[:, permutation]
    permuted_adjacency = adjacency[:, permutation][:, :, permutation]
    permuted = model.encode_target(
        batch["target_latent"][:, permutation],
        permuted_features,
        permuted_adjacency,
        key_mask,
    )
    with_graph.square().mean().backward()
    checks = {
        "graph_changes_output": not torch.allclose(
            with_graph,
            no_graph,
            atol=1.0e-6,
        ),
        "permutation_equivariant": torch.allclose(
            with_graph,
            permuted,
            atol=1.0e-5,
        ),
        "graph_bias_has_gradient": (
            model.graph_bias is not None
            and model.graph_bias.gate.grad is not None
            and float(model.graph_bias.gate.grad.abs()) > 0.0
        ),
        "feature_width_19": features.shape[-1] == 19,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "CPU architecture smoke; no quality claim",
        "checks": checks,
        "blocks": args.blocks,
        "cases": len(indices),
        "feature_width": int(features.shape[-1]),
        "graph_output_delta": float(
            (with_graph - no_graph).abs().max().detach()
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "reuse": {
            "bias": "Trinity GraphBias adapted into floorflow GraphAttentionBias",
            "message_passing": "Trinity MessagePassingMix adapted into floorflow GraphMessageResidual",
            "features": "Trinity conditioning schema adapted from 19-column FEATURE_DIM contract",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
