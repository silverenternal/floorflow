from __future__ import annotations

import torch

from floorflow.data.floorplan_token_targets import (
    build_floorplan_token_vocabulary,
    training_token_sequence,
)
from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    rplan_row_to_canonical,
    rplan_taxonomy,
)
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalRoom,
    canonical_fingerprint,
)
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    UnifiedGeneratorConfig,
    decode_generated_floorplan,
)
from floorflow.representation.polygon_targets import (
    collapse_invalid_polygon_points,
)
from floorflow.representation.unified_floorplan_tokens import (
    decode_floorplan,
    tokenize_floorplan,
    validate_vector_floorplan,
)
from floorflow.training.unified_plan_generator_training import (
    match_candidate_queries,
    unified_generator_loss,
)


def _row() -> dict:
    return {
        "name": "synthetic-rplan",
        "boundary": [
            [10, 10, 1, 1],
            [10, 24, 1, 1],
            [10, 90, 0, 0],
            [90, 90, 3, 0],
            [90, 10, 2, 0],
            [10, 10, 1, 0],
        ],
        "entrance_expand": [
            [10, 10],
            [18, 10],
            [18, 24],
            [10, 24],
        ],
        "rooms": [
            {
                "id": 0,
                "category": 0,
                "location": [40, 50],
                "box": [10, 10, 50, 90],
                "r_boundary": [[10, 10], [50, 10], [50, 90], [10, 90]],
            },
            {
                "id": 1,
                "category": 1,
                "location": [70, 50],
                "box": [50, 10, 90, 90],
                "r_boundary": [[50, 10], [90, 10], [90, 90], [50, 90]],
            },
        ],
        "adjacencies": [[0, 1]],
    }


def _limits() -> RPlanTensorLimits:
    return RPlanTensorLimits(
        image_size=32,
        canvas_size=100,
        max_boundary_points=8,
        max_rooms=4,
        room_points=12,
        max_walls=16,
        max_openings=2,
    )


def test_rplan_adapter_is_token_ready_and_target_free() -> None:
    taxonomy = rplan_taxonomy()
    plan = rplan_row_to_canonical(_row(), taxonomy, image_size=100)
    encoded = tokenize_floorplan(plan, taxonomy)
    decoded = decode_floorplan(encoded.tokens, taxonomy)
    assert len(plan.rooms) == 2
    assert len(plan.openings) == 1
    assert plan.adjacency == (("room_01", "room_02"),)
    assert canonical_fingerprint(decoded) == canonical_fingerprint(encoded.plan)


def test_rplan_collate_counts_cannot_exceed_valid_target_slots() -> None:
    batch = collate_rplan_items(
        [
            {
                "case_name": "truncated",
                "room_exists": torch.tensor([True, False, True]),
                "room_count": torch.tensor(7),
                "wall_valid": torch.tensor([True, False]),
                "opening_valid": torch.tensor([True]),
            }
        ]
    )
    assert batch["target_room_counts"] == [2]
    assert batch["wall_target_counts"] == [1]
    assert batch["opening_target_counts"] == [1]


def test_rplan_dataset_and_generator_gradients() -> None:
    taxonomy = rplan_taxonomy()
    plan = rplan_row_to_canonical(_row(), taxonomy, image_size=100)
    vocabulary, statistics = build_floorplan_token_vocabulary(
        [training_token_sequence(plan, taxonomy)]
    )
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[_row(), _row()],
        limits=_limits(),
        token_vocabulary=vocabulary,
        token_max_length=statistics.recommended_max_length,
    )
    batch = collate_rplan_items([dataset[0], dataset[1]])
    assert batch["boundary_points"].shape == (2, 8, 2)
    assert batch["entrance_points"].shape == (2, 2, 2)
    assert batch["entrance_valid"].tolist() == [True, True]
    assert batch["token_ids"].shape[0] == 2
    assert batch["token_stage_ids"].shape == batch["token_ids"].shape
    assert batch["token_loss_mask"][batch["token_mask"]].all()
    assert torch.allclose(
        batch["entrance_points"][0] * 100,
        torch.tensor([[10.1010, 10.1010], [10.1010, 24.2424]]),
        atol=1.0e-3,
    )
    assert batch["room_polygons"].shape == (2, 4, 12, 2)
    assert batch["room_point_valid"].sum(dim=-1).tolist() == [[4, 4, 0, 0], [4, 4, 0, 0]]
    assert batch["wall_valid"].sum(dim=1).tolist() == [7, 7]
    assert batch["target_room_counts"] == [2, 2]
    assert batch["wall_target_counts"] == [7, 7]
    assert batch["opening_target_counts"] == [1, 1]
    assert batch["junction_points"].shape[:2] == (2, 32)
    assert (batch["junction_valid"].sum(dim=1) > 0).all()
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(
            hidden_dim=16,
            decoder_channels=16,
            identity_queries=4,
            identity_layers=1,
            vector_layers=1,
            polygon_points=12,
            max_walls=16,
            max_openings=2,
            attention_heads=4,
            route_top_k=4,
            max_boundary_points=8,
            enable_point_validity=True,
            use_boundary_edge_graph=True,
            encoder_architecture="compact",
            compact_encoder_channels=(8, 16),
            compact_encoder_blocks=(1, 1),
            enable_program_set=True,
            enable_query_graph_refiner=True,
            enable_entrance_condition=True,
            enable_token_head=True,
            token_vocab_size=12,
            token_max_length=5,
        ),
    )
    base_output = model(
        batch["boundary_mask"],
        batch["boundary_valid"],
        batch["boundary_points"],
        batch["entrance_points"],
        batch["entrance_valid"],
    )
    assert "token_context_tokens" in base_output
    assert "token_logits" not in base_output
    token_ids = torch.randint(0, 12, (2, 5))
    output = model(
        batch["boundary_mask"],
        batch["boundary_valid"],
        batch["boundary_points"],
        batch["entrance_points"],
        batch["entrance_valid"],
        token_ids=token_ids,
    )
    assert output["room_point_valid_logits"].shape == (2, 4, 12)
    assert output["room_query_graph_adjacency"].shape == (2, 4, 4)
    assert output["entrance_condition_token"].shape == (2, 16)
    assert output["token_logits"].shape == (2, 5, 12)
    batch["token_ids"] = token_ids
    batch["token_mask"] = torch.ones_like(token_ids, dtype=torch.bool)
    loss, components = unified_generator_loss(
        output,
        batch,
        taxonomy,
        room_program_weight=0.5,
        token_weight=0.1,
    )
    assert torch.isfinite(loss)
    assert components["total_loss"].ndim == 0
    assert components["canonical_token_ce"] >= 0
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_invalid_polygon_points_collapse_to_valid_vertices() -> None:
    polygons = torch.tensor(
        [
            [
                [
                    [0.1, 0.1],
                    [0.5, 0.1],
                    [0.5, 0.5],
                    [0.1, 0.5],
                    [0.9, 0.9],
                    [0.9, 0.9],
                ]
            ]
        ]
    )
    valid = torch.tensor([[[True, True, True, True, False, False]]])
    collapsed = collapse_invalid_polygon_points(polygons, valid)
    assert torch.equal(collapsed[0, 0, :4], polygons[0, 0, :4])
    assert torch.equal(collapsed[0, 0, 4], polygons[0, 0, 3])
    assert torch.equal(collapsed[0, 0, 5], polygons[0, 0, 3])


def test_invalid_polygon_collapse_preserves_multi_query_last_valid_point() -> None:
    polygons = torch.arange(2 * 2 * 6 * 2, dtype=torch.float32).reshape(2, 2, 6, 2)
    valid = torch.tensor(
        [
            [
                [True, False, True, False, True, False],
                [False, True, True, True, False, False],
            ],
            [
                [True, True, True, True, True, True],
                [False, False, True, True, False, True],
            ],
        ]
    )
    collapsed = collapse_invalid_polygon_points(polygons, valid)
    assert torch.equal(collapsed[0, 0, :3], polygons[0, 0, [0, 2, 4]])
    assert torch.equal(collapsed[0, 0, 3:], polygons[0, 0, 4].expand(3, 2))
    assert torch.equal(collapsed[0, 1, :3], polygons[0, 1, [1, 2, 3]])
    assert torch.equal(collapsed[0, 1, 3:], polygons[0, 1, 3].expand(3, 2))
    assert torch.equal(collapsed[1, 0], polygons[1, 0])
    assert torch.equal(collapsed[1, 1, :3], polygons[1, 1, [2, 3, 5]])
    assert torch.equal(collapsed[1, 1, 3:], polygons[1, 1, 5].expand(3, 2))


def test_cad_query_matching_assigns_positive_supervision_at_initialization() -> None:
    predicted = torch.tensor(
        [
            [
                [[0.1, 0.1], [0.2, 0.1]],
                [[0.3, 0.3], [0.4, 0.3]],
                [[0.7, 0.7], [0.8, 0.7]],
                [[0.8, 0.8], [0.9, 0.8]],
            ]
        ]
    )
    target = torch.tensor(
        [[[[0.9, 0.1], [0.9, 0.2]], [[0.1, 0.8], [0.2, 0.8]], [[0.0, 0.0], [0.0, 0.0]]]]
    )
    pairs, positive = match_candidate_queries(
        predicted,
        target,
        torch.tensor([[True, True, False]]),
        torch.tensor([[False, True, True, False]]),
        target_counts=[2],
    )
    assert len(pairs[0][0]) == 2
    assert positive[0].sum() == 2
    assert set(pairs[0][0].tolist()) == {1, 2}


def test_entrance_condition_preserves_control_initialization() -> None:
    taxonomy = rplan_taxonomy()
    config = {
        "hidden_dim": 16,
        "decoder_channels": 16,
        "identity_queries": 4,
        "identity_layers": 1,
        "vector_layers": 1,
        "polygon_points": 12,
        "max_walls": 16,
        "max_openings": 2,
        "attention_heads": 4,
        "route_top_k": 4,
        "max_boundary_points": 8,
        "encoder_architecture": "compact",
        "compact_encoder_channels": (8, 16),
        "compact_encoder_blocks": (1, 1),
    }
    torch.manual_seed(7)
    control = UnifiedFloorplanGenerator(taxonomy, UnifiedGeneratorConfig(**config))
    control_random = torch.rand(3)
    torch.manual_seed(7)
    conditioned = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(**config, enable_entrance_condition=True),
    )
    conditioned_random = torch.rand(3)
    assert torch.equal(control_random, conditioned_random)
    control_state = control.state_dict()
    for name, value in conditioned.state_dict().items():
        if name.startswith("entrance_geometry."):
            continue
        assert torch.equal(value, control_state[name])


def test_partition_consistent_queries_emit_joint_owner_and_gradients() -> None:
    taxonomy = rplan_taxonomy()
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[_row()],
        limits=_limits(),
    )
    batch = collate_rplan_items([dataset[0]])
    model = UnifiedFloorplanGenerator(
        taxonomy,
        UnifiedGeneratorConfig(
            hidden_dim=16,
            decoder_channels=16,
            identity_queries=4,
            identity_layers=2,
            vector_layers=1,
            polygon_points=12,
            max_walls=16,
            max_openings=2,
            attention_heads=4,
            route_top_k=4,
            max_boundary_points=8,
            encoder_architecture="compact",
            compact_encoder_channels=(8, 16),
            compact_encoder_blocks=(1, 1),
            enable_partition_consistent_queries=True,
        ),
    )
    output = model(
        batch["boundary_mask"],
        batch["boundary_valid"],
        batch["boundary_points"],
        batch["entrance_points"],
        batch["entrance_valid"],
    )
    probability = output["room_instance_partition_probabilities"]

    assert probability.shape == (1, 4, 16, 16)
    assert torch.allclose(
        probability.sum(dim=1),
        torch.ones_like(probability[:, 0]),
    )
    loss, components = unified_generator_loss(
        output,
        batch,
        taxonomy,
        partition_consistency_weight=0.5,
        partition_conflict_only=True,
        partition_label_smoothing=0.05,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["room_mask_partition_query_pixels"] > 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_invalid_generated_room_is_reported_without_geos_crash() -> None:
    taxonomy = rplan_taxonomy()
    plan = CanonicalFloorplan(
        case_id="invalid-room",
        track="rplan_generation",
        width=100,
        height=100,
        footprint=((0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0)),
        rooms=(
            CanonicalRoom(
                id="room_01",
                label="room:Bedroom",
                polygon=(
                    (20.0, 20.0),
                    (80.0, 80.0),
                    (20.0, 80.0),
                    (80.0, 20.0),
                ),
            ),
        ),
        walls=(),
        openings=(),
        furniture=(),
        adjacency=(),
        source_hash="invalid-room",
    )
    errors = validate_vector_floorplan(
        plan,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    assert "room room_01 has invalid hole topology" in errors


def test_mask_partition_readout_decodes_to_canonical_plan() -> None:
    taxonomy = rplan_taxonomy()
    mask_logits = torch.full((1, 4, 32, 32), -8.0)
    mask_logits[0, 0, :, :16] = 8.0
    mask_logits[0, 1, :, 16:] = 8.0
    predictions = {
        "room_exists": torch.tensor([[8.0, 7.0, -8.0, -8.0]]),
        "room_class": torch.tensor(
            [
                [
                    [8.0, -8.0],
                    [-8.0, 8.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ),
        "room_polygons": torch.zeros(1, 4, 12, 2),
        "room_count_logits": torch.tensor([[0.0, 0.0, 12.0, 0.0, 0.0]]),
        "room_instance_mask_logits": mask_logits,
    }
    boundary = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
    )
    plan = decode_generated_floorplan(
        predictions,
        boundary,
        taxonomy,
        case_id="mask-partition",
        canvas_size=100,
        geometry_source="mask_partition",
        mask_gap_policy="hole_fill",
        entrance_points=torch.tensor([[[0.0, 0.0], [0.0, 0.2]]]),
    )
    assert len(plan.rooms) == 2
    assert len(plan.openings) == 1
    assert abs(plan.openings[0].center[0]) < 1.0e-8
    assert abs(plan.openings[0].center[1] - 10.0) < 0.25
    assert {
        operation.kind for operation in plan.operations
    } == {
        "mask_partition_readout",
        "entrance_conditioned_mask_readout",
        "mask_hole_fill",
        "room_clip_precision_guard",
        "cad_stage_exit",
    }
    tokenized = tokenize_floorplan(plan, taxonomy)
    decoded = decode_floorplan(tokenized.tokens, taxonomy)
    assert canonical_fingerprint(plan) == canonical_fingerprint(tokenized.plan)
    assert canonical_fingerprint(plan) == canonical_fingerprint(decoded)


def test_generated_predictions_decode_into_token_contract() -> None:
    taxonomy = rplan_taxonomy()
    points = 12
    polygons = torch.zeros(1, 4, points, 2)
    left = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.1],
            [0.5, 0.9],
            [0.0, 1.0],
        ]
    )
    right = left + torch.tensor([0.4, 0.0])
    for ring, polygon in zip((left, right), polygons[0, :2], strict=True):
        polygon[:4] = ring
        polygon[4:] = ring[-1]
    predictions = {
        "room_exists": torch.tensor([[8.0, 7.0, -8.0, -8.0]]),
        "room_class": torch.tensor(
            [
                [
                    [8.0, -8.0],
                    [-8.0, 8.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        ),
        "room_polygons": polygons,
        "room_count_logits": torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0]]),
    }
    boundary = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
    )
    plan = decode_generated_floorplan(
        predictions,
        boundary,
        taxonomy,
        case_id="generated",
        canvas_size=100,
    )
    tokenized = tokenize_floorplan(plan, taxonomy)
    decoded = decode_floorplan(tokenized.tokens, taxonomy)
    assert len(plan.rooms) == 2
    assert canonical_fingerprint(decoded) == canonical_fingerprint(tokenized.plan)
