#!/usr/bin/env python3
"""Audit ordered CAD actions, independent layers, and candidate graph wiring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.cad_topology_repair import (
    CadTopologyRepairHead,
    apply_bounded_topology_repair,
)
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.heads.opening_host_router import OpeningHostRouter
from floorflow.model.cad_action_graph import PREDECESSORS, STAGES, CadActionGraph
from floorflow.model.cad_furniture_stage import CadFurnitureStage
from floorflow.model.cad_layers import CadLayeredOutput
from floorflow.objectives.candidate_filtration_loss import filtration_descriptor

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cad_action_layer_contract_audit_v2.json",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 5, "Bath": 5}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checks = {}
    expected_order = (
        "envelope",
        "program",
        "rooms",
        "walls",
        "junctions",
        "openings",
        "furniture",
        "validation",
    )
    checks["stage_order"] = tuple(STAGES) == expected_order
    checks["walls_after_rooms"] = set(PREDECESSORS["walls"]) == {
        "envelope",
        "program",
        "rooms",
    }
    checks["openings_after_hosts"] = set(PREDECESSORS["openings"]) == {
        "rooms",
        "walls",
        "junctions",
    }

    graph = CadActionGraph(
        {name: 4 for name in STAGES},
        hidden_dim=16,
        heads=4,
    ).to(device)
    inputs = {
        name: torch.randn(2, 4, device=device)
        for name in STAGES
    }
    inputs["junctions"] = torch.ones(2, 4, device=device)
    first = graph(inputs)
    inputs["junctions"] = torch.zeros(2, 4, device=device)
    second = graph(inputs)
    checks["junction_token_consumed"] = not torch.allclose(
        first.tokens["junctions"],
        second.tokens["junctions"],
    )

    taxonomy = _taxonomy()
    layers = CadLayeredOutput(
        room_logits=torch.randn(1, len(taxonomy.room_ids), 16, 16, device=device),
        wall_logits=torch.randn(1, 1, 16, 16, device=device),
        opening_logits=torch.randn(1, len(taxonomy.opening_ids), 16, 16, device=device),
        furniture_logits=torch.randn(1, len(taxonomy.furniture_ids), 16, 16, device=device),
    )
    rendered = layers.render(
        torch.ones(1, 16, 16, device=device),
        taxonomy,
    )
    checks["independent_room_layer"] = rendered.room_labels.shape[-2:] == (16, 16)
    checks["independent_wall_layer"] = rendered.wall.shape[-2:] == (16, 16)
    checks["independent_opening_layer"] = rendered.opening.shape[-2:] == (16, 16)
    checks["independent_furniture_layer"] = rendered.furniture.shape[-2:] == (16, 16)

    scorer = CadWallCandidateScorer(8, hidden_dim=16, graph_dim=12).to(device)
    features = torch.randn(1, 8, 16, 16, device=device)
    endpoints = torch.tensor(
        [
            [
                [[0.1, 0.1], [0.8, 0.1]],
                [[0.8, 0.1], [0.8, 0.8]],
                [[0.8, 0.8], [0.1, 0.8]],
            ]
        ],
        device=device,
    )
    valid = torch.ones(1, 3, dtype=torch.bool, device=device)
    evidence = torch.tensor([[0.2, 0.5, 0.8]], device=device)
    graph_tokens = torch.randn(1, len(STAGES), 12, device=device)
    output = scorer(features, endpoints, valid, evidence, graph_tokens)
    checks["evidence_prior_anchor"] = torch.allclose(
        output["score_logits"],
        torch.logit(evidence),
        atol=1.0e-5,
    )
    checks["descriptor_head"] = output["predicted_descriptor"].shape == (1, 8)
    checks["phase_budget_heads"] = (
        output["exterior_budget_count"].shape == (1,)
        and output["interior_budget_count"].shape == (1,)
    )
    descriptor = filtration_descriptor(
        endpoints,
        output["score_logits"].sigmoid(),
        valid,
    )
    checks["filtration_descriptor"] = descriptor.shape == (1, 8)
    opening_router = OpeningHostRouter(16, num_types=2).to(device)
    routed = opening_router(
        torch.randn(1, 3, 16, device=device),
        torch.randn(1, 4, 16, device=device),
        torch.tensor(
            [
                [
                    [[0.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 1.0]],
                    [[0.0, 1.0], [1.0, 1.0]],
                ]
            ],
            device=device,
        ),
    )
    checks["opening_host_router"] = routed["host_logits"].shape == (1, 4, 3)
    checks["multiple_openings_per_wall_capacity"] = (
        routed["host_index"].shape == (1, 4)
    )
    furniture_stage = CadFurnitureStage(
        8,
        len(taxonomy.furniture_ids),
        program_channels=4,
        stage_token_dim=8,
        hidden_dim=16,
        max_queries=4,
        min_queries=2,
    ).to(device)
    furniture_controls = {
        "program": torch.randn(1, 4, device=device),
        "room_logits": layers.room_logits.float(),
    }
    furniture_queries = furniture_stage(
        features[:, :8],
        furniture_controls,
        torch.ones(1, 16, 16, device=device),
        stage_token=first.tokens["furniture"][:1, :8],
    )
    checks["furniture_query_decoder"] = (
        furniture_queries["mask_logits"].shape[:2] == (1, 4)
    )
    repair = CadTopologyRepairHead(
        8,
        4,
        hidden_dim=16,
        heads=4,
    ).to(device)
    repair_output = repair(
        torch.randn(1, 3, 8, device=device),
        endpoints,
        valid,
        evidence,
        torch.tensor([[0.2, 0.8, 0.4]], device=device),
        torch.tensor([[[0.8, 0.2, 0.0]]], device=device),
        torch.zeros(1, 6, device=device),
        first.tokens["validation"][:1, :4],
    )
    repaired = apply_bounded_topology_repair(
        endpoints,
        valid,
        repair_output["wall_keep_logits"],
        repair_output["endpoint_delta"],
        max_edits=1,
    )
    checks["validation_repair_consumer"] = (
        repair_output["wall_keep_logits"].shape == (1, 3)
    )
    checks["bounded_graph_edits"] = (
        bool(repaired["edited"].sum(dim=1).max() <= 1)
    )
    checks["all_finite"] = all(
        bool(torch.isfinite(value).all())
        for value in output.values()
        if torch.is_tensor(value)
    )
    checks["all_finite"] = checks["all_finite"] and all(
        bool(torch.isfinite(value).all())
        for value in furniture_queries.values()
        if torch.is_tensor(value)
    )
    checks["all_finite"] = checks["all_finite"] and all(
        bool(torch.isfinite(value).all())
        for value in repair_output.values()
        if torch.is_tensor(value)
    )

    report = {
        "schema_version": "cad-action-layer-contract-audit-v2",
        "device": str(device),
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": all(checks.values()),
        "stage_predecessors": {
            key: list(value)
            for key, value in PREDECESSORS.items()
        },
        "documented_map": str(
            ROOT / "docs/predicted_control_cad_action_layer_map_v2.md"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
