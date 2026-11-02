from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.cad_topology_repair import (
    CadTopologyRepairHead,
    apply_bounded_topology_repair,
)
from floorflow.heads.vector_primitive_head import VectorPrimitiveHead
from floorflow.inference.cad_wall_candidate_runtime import (
    compose_scored_candidate_plan,
)
from floorflow.model.cad_action_graph import STAGES, CadActionGraph
from floorflow.model.cad_furniture_stage import CadFurnitureStage
from floorflow.model.official_mita_route import OfficialMiTARouteAttention


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 5}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_furniture_stage_is_program_and_action_conditioned() -> None:
    taxonomy = _taxonomy()
    stage = CadFurnitureStage(
        8,
        len(taxonomy.furniture_ids),
        program_channels=3,
        stage_token_dim=8,
        hidden_dim=16,
        max_queries=4,
        min_queries=2,
        num_rooms=len(taxonomy.room_ids),
        enable_room_assignment=True,
        enable_semantic_router=True,
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    controls = {
        "program": torch.randn(2, 3),
        "room_logits": torch.randn(2, len(taxonomy.room_ids), 16, 16),
    }
    footprint = torch.ones(2, 16, 16)
    first = stage(
        features,
        controls,
        footprint,
        stage_token=torch.zeros(2, 8),
    )
    second = stage(
        features,
        controls,
        footprint,
        stage_token=torch.ones(2, 8),
    )
    assert first["mask_logits"].shape[1] == 4
    assert first["class_logits"].shape[-1] == len(taxonomy.furniture_ids) + 1
    assert first["room_assignment_logits"].shape == (
        2,
        4,
        len(taxonomy.room_ids) + 1,
    )
    assert first["furniture_semantic_gate"].shape == (
        2,
        4,
        len(taxonomy.furniture_ids),
    )
    loss = (
        first["mask_logits"].sum()
        + first["class_logits"].sum()
        + second["class_logits"].sum()
    )
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(stage.stage_film.weight.grad) > 0
    assert torch.count_nonzero(stage.program[0].weight.grad) > 0
    assert torch.count_nonzero(stage.room_projection[0].weight.grad) > 0
    assert torch.count_nonzero(stage.head.room_fusion.weight.grad) > 0
    assert stage.room_assignment.weight.grad is not None
    assert torch.count_nonzero(
        stage.semantic_router.query_projection[0].weight.grad
    ) > 0
    assert torch.count_nonzero(
        stage.semantic_router.room_projection[0].weight.grad
    ) > 0
    assert torch.count_nonzero(
        stage.semantic_router.gate[0].weight.grad
    ) > 0


def test_repair_consumes_final_diagnostics_and_is_bounded() -> None:
    head = CadTopologyRepairHead(
        8,
        6,
        hidden_dim=16,
        heads=4,
        max_endpoint_delta=0.02,
    )
    endpoints = torch.tensor(
        [
            [
                [[0.1, 0.1], [0.8, 0.1]],
                [[0.8, 0.1], [0.8, 0.8]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    valid = torch.tensor([[True, True, False]])
    opening_attachment = torch.tensor(
        [
            [
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
            ]
        ]
    )
    output = head(
        torch.randn(1, 3, 8),
        endpoints,
        valid,
        torch.tensor([[0.8, 0.5, 0.0]]),
        torch.tensor([[0.2, 0.9, 0.0]]),
        opening_attachment,
        torch.zeros(1, 6),
        torch.randn(1, 6),
    )
    assert output["wall_keep_logits"].shape == (1, 3)
    assert output["endpoint_delta"].abs().max() <= 0.02 + 1.0e-6
    assert output["opening_host_logits"].shape == (1, 2, 3)
    assert torch.isfinite(output["opening_host_logits"]).all()
    loss = sum(
        value.float().sum()
        for value in output.values()
    )
    loss.backward()
    assert head.host_residual_scale.grad is not None

    keep = torch.tensor([[9.0, -9.0, 9.0]])
    delta = torch.zeros(1, 3, 2, 2)
    delta[0, 1, 0] = 0.2
    delta[0, 0, 1] = 0.2
    bounded = apply_bounded_topology_repair(
        endpoints,
        valid,
        keep,
        delta,
        max_edits=1,
    )
    assert bounded["edited"].sum(dim=1).max() <= 1
    assert bounded["valid"].tolist() == [[True, False, False]]


def test_cad_action_order_remains_production_order() -> None:
    assert STAGES == (
        "envelope",
        "program",
        "rooms",
        "walls",
        "junctions",
        "openings",
        "furniture",
        "validation",
    )


def test_cad_action_graph_propagates_room_changes_downstream() -> None:
    graph = CadActionGraph(
        {
            "envelope": 1,
            "program": 2,
            "rooms": 3,
            "walls": 1,
            "junctions": 1,
            "openings": 3,
            "furniture": 2,
            "validation": 2,
        },
        hidden_dim=8,
        heads=4,
    )
    controls = {
        "envelope": torch.ones(1, 1, 16, 16),
        "program": torch.randn(1, 2),
        "rooms": torch.randn(1, 3, 8, 8),
        "walls": torch.randn(1, 1, 8, 8),
        "junctions": torch.randn(1, 1, 8, 8),
        "openings": torch.randn(1, 3, 8, 8),
        "furniture": torch.randn(1, 2, 8, 8),
        "validation": torch.randn(1, 2, 8, 8),
    }
    first = graph(controls)
    controls["rooms"] = torch.randn_like(controls["rooms"])
    second = graph(controls)
    for stage in ("walls", "junctions", "openings", "furniture"):
        assert not torch.allclose(
            first.tokens[stage],
            second.tokens[stage],
        )


def test_soft_router_budget_has_a_differentiable_attention_effect() -> None:
    torch.manual_seed(7)
    route = OfficialMiTARouteAttention(
        dim=8,
        num_heads=2,
        router_count=2,
        top_k=2,
    )
    query = torch.randn(1, 2, 8)
    memory = torch.randn(1, 2, 8)
    low_budget = torch.full((1, 2), 1.0, requires_grad=True)
    high_budget = torch.full((1, 2), 2.0)
    low = route(
        query,
        memory,
        router_budgets=low_budget.detach().long(),
        router_soft_budgets=low_budget,
    )[0]
    high = route(
        query,
        memory,
        router_budgets=high_budget.long(),
        router_soft_budgets=high_budget,
    )[0]
    assert not torch.allclose(low, high)
    low.square().mean().backward()
    assert low_budget.grad is not None
    assert torch.count_nonzero(low_budget.grad) > 0


def test_vector_stages_receive_their_ordered_action_tokens() -> None:
    head = VectorPrimitiveHead(
        8,
        room_classes=2,
        opening_classes=1,
        hidden_dim=16,
        max_rooms=3,
        max_walls=5,
        max_openings=2,
        polygon_points=4,
        decoder_layers=1,
        stage_context=True,
        ordered_stage_context=True,
    )
    stage_tokens = {
        "walls": torch.randn(2, 16, requires_grad=True),
        "junctions": torch.randn(2, 16),
        "openings": torch.randn(2, 16),
    }

    output = head(
        torch.randn(2, 8, 16, 16),
        stage_tokens=stage_tokens,
    )
    loss = (
        output["wall_endpoints"].square().mean()
        + output["opening_centers"].square().mean()
    )
    loss.backward()

    wall_weight = head.stage_conditioners["wall"][-1].weight
    opening_weight = head.stage_conditioners["opening"][-1].weight
    assert wall_weight.grad is not None
    assert opening_weight.grad is not None
    assert torch.count_nonzero(wall_weight.grad) > 0
    assert torch.count_nonzero(opening_weight.grad) > 0


def test_vector_stage_context_is_opt_in_and_fails_closed() -> None:
    legacy = VectorPrimitiveHead(
        8,
        room_classes=2,
        opening_classes=1,
        hidden_dim=16,
        max_rooms=2,
        max_walls=3,
        max_openings=1,
    )
    assert legacy.stage_conditioners is None

    ordered = VectorPrimitiveHead(
        8,
        room_classes=2,
        opening_classes=1,
        hidden_dim=16,
        max_rooms=2,
        max_walls=3,
        max_openings=1,
        ordered_stage_context=True,
    )
    with np.testing.assert_raises_regex(
        ValueError,
        "requires stage tokens",
    ):
        ordered(torch.randn(1, 8, 16, 16))


def test_bounded_repair_is_consumed_by_candidate_plan_composition() -> None:
    taxonomy = _taxonomy()
    endpoints = torch.tensor(
        [
            [[0.1, 0.1], [0.8, 0.1]],
            [[0.8, 0.1], [0.8, 0.8]],
        ]
    )
    delta = torch.zeros(2, 2, 2)
    delta[0, 0, 0] = 0.01
    plan = compose_scored_candidate_plan(
        endpoints,
        torch.tensor([0.9, 0.8]),
        torch.ones(2, dtype=torch.bool),
        [],
        32.0,
        [],
        np.zeros((2, 32, 32), dtype=np.float32),
        taxonomy,
        score_threshold=0.0,
        repair_keep_logits=torch.tensor([9.0, -9.0]),
        repair_endpoint_delta=delta,
        repair_max_edits=2,
    )
    assert len(plan.walls) == 1
    assert abs(plan.walls[0].start[0] - 0.11 * 32.0) < 1.0e-5
    assert abs(plan.walls[0].start[1] - 0.10 * 32.0) < 1.0e-5


def test_candidate_plan_junctions_use_final_local_wall_indices() -> None:
    taxonomy = _taxonomy()
    endpoints = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    plan = compose_scored_candidate_plan(
        endpoints,
        torch.tensor([0.1, 0.9]),
        torch.ones(2, dtype=torch.bool),
        [],
        32.0,
        [],
        np.zeros((2, 32, 32), dtype=np.float32),
        taxonomy,
        score_threshold=0.5,
    )
    assert len(plan.walls) == 1
    assert plan.junctions
    assert all(
        junction["incident_walls"] == [0]
        for junction in plan.junctions
    )
