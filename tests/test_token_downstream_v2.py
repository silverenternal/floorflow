from __future__ import annotations

import torch
from torch import nn

from floorflow.data.length_bucketing import LengthBucketBatchSampler
from floorflow.data.rplan_boundary_adapter import collate_rplan_items
from floorflow.evaluation.token_readout import token_teacher_forced_metrics
from floorflow.heads.floorplan_token_head import FloorplanTokenHead
from floorflow.heads.token_vector_refiner import TokenVectorRefiner
from floorflow.inference.token_refined_inference import token_refined_generation
from floorflow.objectives.gradient_conflict import conflict_aware_backward
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary
from floorflow.objectives.token_soft_graph import token_soft_graph_consistency_loss
from floorflow.objectives.token_structure_consistency import (
    token_structure_consistency_loss,
)
from floorflow.representation.unified_floorplan_tokens import FloorplanToken
from floorflow.training.token_consumer_validation import (
    aggregate_token_consumer_reports,
)
from floorflow.training.token_optimizer import build_floorflow_optimizer


def test_dynamic_collate_and_length_bucketing() -> None:
    items = [
        {
            "case_name": "a",
            "token_ids": torch.tensor([1, 2, 3]),
            "token_mask": torch.ones(3, dtype=torch.bool),
            "token_loss_mask": torch.ones(3, dtype=torch.bool),
            "token_stage_ids": torch.zeros(3, dtype=torch.long),
            "token_length": torch.tensor(3),
            "token_oov_count": torch.tensor(0),
        },
        {
            "case_name": "b",
            "token_ids": torch.tensor([4, 5, 6, 7]),
            "token_mask": torch.ones(4, dtype=torch.bool),
            "token_loss_mask": torch.ones(4, dtype=torch.bool),
            "token_stage_ids": torch.ones(4, dtype=torch.long),
            "token_length": torch.tensor(4),
            "token_oov_count": torch.tensor(0),
        },
    ]
    batch = collate_rplan_items(items)
    assert batch["token_ids"].shape == (2, 4)
    assert batch["token_ids"][0, -1] == -1
    assert len(batch["verifier_target_keys"]) == 2
    assert batch["verifier_target_keys"][0] != batch["verifier_target_keys"][1]
    sampler = LengthBucketBatchSampler([9, 1, 5, 3], 2, seed=3)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert len(sampler) == 2


def test_token_vector_refiner_updates_all_geometry() -> None:
    refiner = TokenVectorRefiner(
        8,
        max_walls=2,
        max_openings=2,
        polygon_points=3,
        heads=4,
        residual_scale=0.02,
    )
    predictions = {
        "room_features": torch.randn(1, 2, 8),
        "room_polygons": torch.zeros(1, 2, 3, 2),
        "wall_endpoints": torch.zeros(1, 2, 2, 2),
        "opening_centers": torch.zeros(1, 2, 2),
        "opening_widths": torch.zeros(1, 2),
    }
    refined = refiner(
        predictions,
        torch.randn(1, 5, 8),
        torch.tensor([[True, True, True, False, False]]),
    )
    assert refined["room_polygons"].shape == (1, 2, 3, 2)
    assert refined["wall_endpoints"].shape == (1, 2, 2, 2)
    assert refined["token_refinement_applied"].item() == 1.0
    assert 0.0 <= refined["token_refiner_gate_room"].item() <= 1.0
    assert refined["token_refiner_room_residual_norm"] >= 0.0


def test_typed_graph_consistency_components() -> None:
    loss, components = token_structure_consistency_loss(
        torch.tensor([[1, 2, 3]]),
        torch.ones(1, 3, dtype=torch.bool),
        FloorplanTokenVocabulary(("ROOM_BEGIN|x", "WALL|x", "OPENING|x")),
        room_exists_logits=torch.zeros(1, 2),
        wall_exists_logits=torch.zeros(1, 2),
        opening_exists_logits=torch.zeros(1, 1),
        room_mask_logits=torch.zeros(1, 2, 4, 4),
        target_valid=torch.tensor([[True, True]]),
        adjacency_logits=torch.zeros(1, 2, 2),
        target_adjacency=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        room_class_logits=torch.zeros(1, 2, 3),
        target_room_classes=torch.tensor([[0, 1]]),
        wall_exterior_logits=torch.zeros(1, 2),
        target_wall_exterior=torch.tensor([[1.0, 0.0]]),
        wall_valid=torch.ones(1, 2, dtype=torch.bool),
        opening_host_logits=torch.zeros(1, 1, 2),
        target_opening_host=torch.tensor([[1]]),
        opening_valid=torch.ones(1, 1, dtype=torch.bool),
        room_point_valid_logits=torch.zeros(1, 2, 3),
        target_room_point_valid=torch.ones(1, 2, 3, dtype=torch.bool),
        room_polygons=torch.zeros(1, 2, 3, 2),
        target_room_polygons=torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]] * 2]
        ),
        wall_endpoints=torch.zeros(1, 2, 2, 2),
        junction_points=torch.tensor([[[0.0, 0.0]]]),
        target_junction_degree=torch.tensor([[2]]),
        junction_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    for key in (
        "token_adjacency_consistency",
        "token_wall_type_consistency",
        "token_opening_host_consistency",
        "token_room_label_consistency",
        "token_ring_vertex_consistency",
        "token_ring_closure_consistency",
        "token_junction_degree_consistency",
    ):
        assert key in components


def test_conflict_backward_projects_opposing_gradient() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    direction = torch.tensor([1.0, 0.0])
    primary = torch.dot(parameter, direction).square()
    conflict = -torch.dot(parameter, direction).square()
    stats = conflict_aware_backward(
        {"primary": primary, "conflict": conflict},
        [parameter],
        (("primary", "conflict"),),
    )
    assert parameter.grad is not None
    assert abs(float(parameter.grad[0]) + 2.0) < 1.0e-6
    assert stats["primary__conflict__projected"] == 1.0


def test_incremental_conflict_backward_preserves_total_objective() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    direction = torch.tensor([1.0, 0.0])
    primary = torch.dot(parameter, direction).square()
    conflict = -torch.dot(parameter, direction).square()
    auxiliary = parameter.square().sum()
    losses = {
        "primary": primary,
        "conflict": conflict,
        "auxiliary": auxiliary,
    }
    stats = conflict_aware_backward(
        losses,
        [parameter],
        (("primary", "conflict"),),
        total_loss=primary + conflict + auxiliary,
    )
    expected = torch.tensor([0.0, 4.0])
    assert parameter.grad is not None
    assert torch.allclose(parameter.grad, expected)
    assert stats["primary__conflict__projected"] == 1.0


def test_token_optimizer_and_schedule() -> None:
    model = nn.Module()
    model.backbone = nn.Linear(4, 4)
    model.token_head = nn.Linear(4, 4)
    model.token_norm = nn.LayerNorm(4)
    optimizer = build_floorflow_optimizer(
        model,
        learning_rate=2.0e-4,
        weight_decay=1.0e-4,
        token_learning_rate_multiplier=2.0,
    )
    token_groups = [
        group for group in optimizer.param_groups if group["lr"] == 4.0e-4
    ]
    assert token_groups


def test_token_readout_reports_kind_and_stage_metrics() -> None:
    vocabulary = FloorplanTokenVocabulary(("ROOM_BEGIN|x", "WALL|x", "END"))
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    metrics = token_teacher_forced_metrics(
        logits,
        torch.tensor([[0, 1, 2]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[2, 3, 7]]),
        vocabulary,
    )
    assert metrics["token_accuracy"] == 1.0
    assert metrics["token_macro_f1"] == 1.0
    assert metrics["token_room_count_abs_error"] == 0.0


def test_token_readout_ignores_dynamic_padding_ids() -> None:
    vocabulary = FloorplanTokenVocabulary(("ROOM_BEGIN|x", "WALL|x", "END"))
    logits = torch.zeros(2, 4, vocabulary.model_vocab_size)
    logits[0, 0, 0] = 10.0
    logits[1, 0, 1] = 10.0
    target_ids = torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]])
    valid = torch.tensor(
        [[True, False, False, False], [True, False, False, False]]
    )
    stage_ids = torch.zeros_like(target_ids)
    metrics = token_teacher_forced_metrics(
        logits,
        target_ids,
        valid,
        stage_ids,
        vocabulary,
    )
    assert metrics["token_accuracy"] == 1.0
    assert metrics["token_teacher_forced_end_accuracy"] == 0.0


def test_two_pass_token_refined_inference_consumes_generated_tokens() -> None:
    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_head = nn.Identity()

        def forward(self, *args, token_ids=None, **kwargs):
            if token_ids is None:
                return {
                    "token_context_tokens": torch.zeros(1, 1, 4),
                    "token_context_valid": torch.ones(1, 1, dtype=torch.bool),
                }
            return {
                "token_refinement_applied": torch.tensor(1.0),
                "token_ids": token_ids,
            }

        def refine_with_token_ids(self, output, token_ids, *args):
            return {
                **output,
                "token_refinement_applied": torch.tensor(1.0),
                "token_ids": token_ids,
            }

    vocabulary = FloorplanTokenVocabulary(("END",))
    base, refined, generated = token_refined_generation(
        FakeModel(),
        torch.zeros(1, 1, 4, 4),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2),
        torch.ones(1, dtype=torch.bool),
        vocabulary,
        (FloorplanToken("END"),),
        max_new_tokens=1,
    )
    assert "token_logits" not in base[0]
    assert refined[0]["token_refinement_applied"].item() == 1.0
    assert generated[0][-1].kind == "END"


def test_token_logits_imply_soft_graph_consistency() -> None:
    tokens = (
        "ADJACENCY|room_01|room_02",
        "ROOM_BEGIN|room_01|room:LivingRoom|<REF>",
        "WALL|wall_0001|<REF>|<REF>|<REF>|<REF>|<REF>|exterior",
        "WALL|wall_0002|<REF>|<REF>|<REF>|<REF>|<REF>|interior",
        "OPENING|opening_0001|door:Door|wall_0001|<REF>|<REF>|<REF>|<REF>",
    )
    vocabulary = FloorplanTokenVocabulary(tokens)
    logits = torch.full((1, 4, vocabulary.model_vocab_size), -8.0)
    for position, token_id in enumerate((0, 1, 2, 4)):
        logits[0, position, token_id] = 8.0
    predictions = {
        "room_adjacency_logits": torch.zeros(1, 2, 2),
        "room_class": torch.zeros(1, 2, 1),
        "wall_exterior_logits": torch.zeros(1, 2),
        "opening_host_logits": torch.zeros(1, 1, 2),
    }
    loss, components = token_soft_graph_consistency_loss(
        logits,
        vocabulary,
        predictions,
        token_valid=torch.ones(1, 4, dtype=torch.bool),
        room_valid=torch.ones(1, 2, dtype=torch.bool),
        wall_valid=torch.ones(1, 2, dtype=torch.bool),
        opening_valid=torch.ones(1, 1, dtype=torch.bool),
        room_labels=("room:LivingRoom",),
    )
    assert torch.isfinite(loss)
    assert len(components) == 4


def test_incremental_token_cache_matches_full_causal_forward() -> None:
    torch.manual_seed(5)
    head = FloorplanTokenHead(8, 12, max_length=8, hidden_dim=8)
    head.eval()
    context = torch.randn(1, 3, 8)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    token_ids = torch.tensor([[1, 2, 3, 4]])
    stages = torch.tensor([[0, 2, 2, 7]])
    full = head(context, token_ids, None, stages, context_mask)
    cache = None
    incremental = []
    inputs = [head.bos_id, 1, 2, 3]
    for position, token_id in enumerate(inputs):
        logits, cache = head.incremental_step(
            context,
            context_mask,
            torch.tensor([token_id]),
            stages[:, position],
            cache,
        )
        incremental.append(logits)
    stacked = torch.stack(incremental, dim=1)
    assert torch.allclose(full, stacked, atol=1.0e-5, rtol=1.0e-4)


def test_consumer_bootstrap_aggregation() -> None:
    def report(vector_valid: bool, generated_valid: bool):
        details = {
            "vector": [{"contract_valid": vector_valid, "functional_watertight": vector_valid}],
            "teacher": [{"contract_valid": vector_valid, "functional_watertight": vector_valid}],
            "generated": [
                {
                    "contract_valid": generated_valid,
                    "functional_watertight": generated_valid,
                }
            ],
        }
        metrics = {"token_consumer_generation_completed": float(generated_valid)}
        return metrics, details

    aggregated = aggregate_token_consumer_reports(
        [report(True, False), report(False, False)],
        bootstrap_iterations=20,
    )
    assert aggregated["token_consumer_cases"] == 2.0
    assert aggregated["token_consumer_generated_minus_vector_valid"] == -0.5
    assert "token_consumer_generated_minus_vector_valid_ci_low" in aggregated
