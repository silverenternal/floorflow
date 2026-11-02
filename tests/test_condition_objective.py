from __future__ import annotations

from collections import Counter

import torch
import pytest

from floorflow.objectives.condition_objective import (
    ConditionObjectiveWeights,
    condition_layout_loss,
    partition_boundary_regularizer,
    partition_room_labels,
    room_macro_iou_from_labels,
)
from floorflow.evaluation.condition_evaluation import case_metrics
from floorflow.objectives.loss_audit import (
    ObjectiveTerm,
    compose_objective,
    contribution_rows,
    gradient_contribution_rows,
)
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.heads.room_partition import batch_room_partition_target
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _model(
    taxonomy,
    hierarchical_rooms: bool = False,
    polygon_loop_head: bool = False,
    room_mask_head: bool = False,
):
    return FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        dynamic_output_head=False,
        topology_aux=False,
        semantic_endpoint_head=True,
        num_rooms=len(taxonomy.room_ids),
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
        room_partition_semantic_prior=True,
        room_class_ids=tuple(taxonomy.room_ids),
        structured_object_head=True,
        semantic_taxonomy=taxonomy,
        hierarchical_rooms=hierarchical_rooms,
        hierarchical_hidden=16,
        hierarchical_tokens=2,
        hierarchical_coarse_resolution=8,
        polygon_loop_head=polygon_loop_head,
        polygon_queries=6,
        polygon_points=12,
        polygon_hidden=24,
        room_mask_head=room_mask_head,
        room_mask_queries=6,
        room_mask_resolution=8,
    )


def test_dense_partition_is_primary_and_component_gradient_is_auditable() -> None:
    torch.manual_seed(3)
    taxonomy = _taxonomy()
    model = _model(taxonomy)
    target = torch.zeros(2, 32, 32, dtype=torch.long)
    bedroom = taxonomy.class_id("room:Bedroom")
    kitchen = taxonomy.class_id("room:Kitchen")
    wall = taxonomy.class_id("wall")
    door = taxonomy.class_id("door:Swing Beside")
    target[:, 4:28, 4:16] = bedroom
    target[:, 4:28, 16:28] = kitchen
    target[:, 4:28, 15:17] = wall
    target[:, 10:12, 15:17] = door
    footprint = torch.zeros(2, 32, 32)
    footprint[:, 4:28, 4:28] = 1.0
    zero = torch.zeros(2, model.target_channels, 32, 32)
    _, aux = model.forward_with_aux(zero, torch.zeros(2), footprint)
    loss, components, terms = condition_layout_loss(
        aux,
        target,
        footprint,
        taxonomy,
        weights=ConditionObjectiveWeights(),
    )
    assert torch.isfinite(loss)
    assert {"room_partition", "room_boundary", "opening", "furniture", "wall"}.issubset(components)
    rows = contribution_rows(terms)
    assert sum(row["absolute_share"] for row in rows) == pytest.approx(1.0)
    gradient_rows = gradient_contribution_rows(terms, model.parameters(), primary="room_partition")
    assert all(torch.isfinite(torch.tensor(row["gradient_norm"])) for row in gradient_rows)
    loss.backward()
    dense_target = batch_room_partition_target(target, footprint, taxonomy)
    assert (dense_target[footprint.bool()] >= 0).all()
    labels = partition_room_labels(aux["room_partition_logits"], taxonomy)
    assert 0.0 <= room_macro_iou_from_labels(labels, target, taxonomy) <= 1.0


def test_partition_boundary_loss_handles_single_room() -> None:
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    loss = partition_boundary_regularizer(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_objective_weights_are_explicit() -> None:
    left = torch.tensor(2.0, requires_grad=True)
    right = torch.tensor(3.0, requires_grad=True)
    total, components = compose_objective(
        [ObjectiveTerm("primary", left, 1.0), ObjectiveTerm("aux", right, 0.25)]
    )
    assert total.item() == 2.75
    assert set(components) == {"primary", "aux", "loss"}


def test_object_gradient_isolation_preserves_shared_backbone() -> None:
    taxonomy = _taxonomy()
    model = _model(taxonomy)
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 4:28, 4:16] = taxonomy.class_id("room:Bedroom")
    target[:, 4:28, 16:28] = taxonomy.class_id("room:Kitchen")
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1.0
    _, aux = model.forward_with_aux(
        torch.zeros(1, model.target_channels, 32, 32),
        torch.zeros(1),
        footprint,
        detach_object_features=True,
    )
    _, _, terms = condition_layout_loss(aux, target, footprint, taxonomy)
    object_term = next(term for term in terms if term.name == "opening")
    backbone_gradient = torch.autograd.grad(
        object_term.weighted_value,
        model.condition_encoder.stem.weight,
        allow_unused=True,
        retain_graph=True,
    )[0]
    head_gradient = torch.autograd.grad(
        object_term.weighted_value,
        model.structured_objects.opening.weight,
        allow_unused=True,
    )[0]
    assert backbone_gradient is None or torch.count_nonzero(backbone_gradient) == 0
    assert head_gradient is not None and torch.count_nonzero(head_gradient) > 0


def test_hierarchical_group_head_shapes() -> None:
    taxonomy = _taxonomy()
    model = FloorplanFlowModel(
        target_channels=taxonomy.num_target_channels,
        base_channels=8,
        levels=2,
        condition_dropout=0.0,
        dynamic_output_head=False,
        topology_aux=False,
        semantic_endpoint_head=True,
        num_rooms=len(taxonomy.room_ids),
        room_partition_head=True,
        num_room_classes=len(taxonomy.room_ids),
        room_partition_semantic_prior=True,
        room_class_ids=tuple(taxonomy.room_ids),
        room_group_head=True,
        semantic_taxonomy=taxonomy,
    )
    footprint = torch.ones(1, 32, 32)
    _, aux = model.forward_with_aux(torch.zeros(1, model.target_channels, 32, 32), torch.zeros(1), footprint)
    assert aux["room_group_logits"].shape == (1, 8, 32, 32)


def test_hierarchical_coarse_refine_and_transport_loss_backpropagate() -> None:
    taxonomy = _taxonomy()
    model = _model(
        taxonomy,
        hierarchical_rooms=True,
        polygon_loop_head=True,
        room_mask_head=True,
    )
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 4:28, 4:16] = taxonomy.class_id("room:Bedroom")
    target[:, 4:28, 16:28] = taxonomy.class_id("room:Kitchen")
    footprint = torch.zeros(1, 32, 32)
    footprint[:, 4:28, 4:28] = 1.0
    _, aux = model.forward_with_aux(
        torch.zeros(1, model.target_channels, 32, 32),
        torch.zeros(1),
        footprint,
    )
    loss, components, terms = condition_layout_loss(
        aux,
        target,
        footprint,
        taxonomy,
        weights=ConditionObjectiveWeights(
            room_hierarchy=0.5,
            room_refine=0.25,
            room_transport=0.2,
            room_polygon=0.1,
            room_mask=0.2,
            room_mask_partition=0.2,
        ),
    )
    assert {
        "room_hierarchy",
        "room_refine",
        "room_transport_loss",
        "room_polygon_loss",
        "room_mask_objective",
    }.issubset(components)
    assert any(term.name == "room_hierarchy" for term in terms)
    assert any(term.name == "room_polygon" for term in terms)
    assert any(term.name == "room_mask" for term in terms)
    loss.backward()
    grad = model.hierarchical_rooms.refiner.residual_scale.grad
    assert grad is not None and torch.count_nonzero(grad) > 0
    decoded = decode_structured_floorplan(
        model,
        target,
        footprint,
        taxonomy,
        room_source="hierarchical",
        object_source="heads",
    )
    assert decoded.shape == target.shape
    assert torch.isin(decoded, torch.tensor(taxonomy.room_ids)).any()
    semantic_decoded = decode_structured_floorplan(
        model,
        target,
        footprint,
        taxonomy,
        room_source="semantic",
        object_source="heads",
    )
    assert semantic_decoded.shape == target.shape
    union_decoded = decode_structured_floorplan(
        model,
        target,
        footprint,
        taxonomy,
        room_source="partition",
        object_source="heads",
        structural_gate=True,
        wall_mode="union",
    )
    assert union_decoded.shape == target.shape
    validation_metrics = case_metrics(aux, target, footprint, taxonomy)
    assert "refined_room_iou" in validation_metrics
    assert "refined_room_grid_iou" in validation_metrics
