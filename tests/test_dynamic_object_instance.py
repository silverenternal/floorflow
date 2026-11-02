from __future__ import annotations

import pytest
import torch

from floorflow.heads.dynamic_object_instance import (
    DynamicFurnitureInstanceHead,
    DynamicWallOpeningIntervalHead,
    object_instance_targets,
    render_object_queries,
    render_object_queries_soft,
    wall_endpoints_from_masks,
)


def test_dynamic_furniture_head_returns_query_budget_and_masks() -> None:
    head = DynamicFurnitureInstanceHead(
        8,
        3,
        hidden_dim=16,
        max_queries=6,
        min_queries=2,
    )
    features = torch.randn(2, 8, 16, 16)
    room_probability = torch.full((2, 1, 16, 16), 0.5)
    footprint = torch.ones(2, 1, 16, 16)
    output = head(features, room_probability, footprint)
    assert output["mask_logits"].shape[:2] == (2, 6)
    assert output["class_logits"].shape[:2] == (2, 6)
    assert (output["query_budget"] >= 2).all()
    assert (output["query_budget"] <= 6).all()
    with_rooms = head(
        features,
        room_probability,
        footprint,
        room_tokens=torch.randn(2, 3, 16),
    )
    assert with_rooms["mask_logits"].shape == output["mask_logits"].shape
    assert not torch.equal(
        with_rooms["class_logits"],
        output["class_logits"],
    )
    class_density = torch.zeros(2, 3, 16, 16)
    class_density[:, 1, 4:8, 4:8] = 0.9
    with torch.no_grad():
        head.decoder.fusion_scale.fill_(0.5)
    with_density = head(
        features,
        room_probability,
        footprint,
        class_density=class_density,
        room_tokens=torch.zeros(2, 3, 16),
    )
    without_density = head(
        features,
        room_probability,
        footprint,
        room_tokens=torch.zeros(2, 3, 16),
    )
    assert with_density["mask_class_evidence"] is not None
    assert not torch.equal(
        with_density["class_logits"],
        without_density["class_logits"],
    )
    assert (with_density["query_budget"] >= without_density["query_budget"]).all()
    with pytest.raises(ValueError, match="class_density"):
        head(
            features,
            room_probability,
            footprint,
            class_density=torch.zeros(2, 2, 16, 16),
        )

    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[0, 2:6, 2:6] = 3
    labels[1, 8:12, 8:12] = 4
    classes, masks, valid = object_instance_targets(
        labels,
        (3, 4),
        max_instances=6,
    )
    assert classes.shape == masks.shape[:2] == valid.shape == (2, 6)
    assert valid.sum(dim=1).tolist() == [1, 1]

    class_logits = torch.full((1, 2, 4), -5.0)
    class_logits[0, :, 0] = 5.0
    query_masks = torch.full((1, 2, 16, 16), -100.0)
    query_masks[0, 0, 2:6, 2:6] = 8.0
    query_masks[0, 1, 8:12, 8:12] = 8.0
    rendered = render_object_queries(
        class_logits,
        query_masks,
        (3, 4),
        torch.ones(1, 1, 16, 16),
    )
    assert rendered.shape == (1, 16, 16)
    assert (rendered == 3).any()
    rendered_3d = render_object_queries(
        class_logits,
        query_masks,
        (3, 4),
        torch.ones(1, 16, 16),
    )
    assert torch.equal(rendered, rendered_3d)
    active = torch.tensor([[True, False]])
    active_only = render_object_queries(
        class_logits,
        query_masks,
        (3, 4),
        torch.ones(1, 1, 16, 16),
        active_queries=active,
    )
    assert (active_only[0, 2:6, 2:6] == 3).all()
    assert active_only[0, 8:12, 8:12].eq(0).all()
    soft_active = render_object_queries_soft(
        class_logits,
        query_masks,
        torch.ones(1, 1, 16, 16),
        active_queries=active,
    )
    assert soft_active[0, 1, 8:12, 8:12].eq(0).all()
    with pytest.raises(ValueError, match="active query"):
        render_object_queries(
            class_logits,
            query_masks,
            (3, 4),
            torch.ones(1, 1, 16, 16),
            active_queries=torch.zeros(1, 2, dtype=torch.bool),
        )


def test_wall_interval_head_produces_attached_intervals() -> None:
    wall_masks = torch.full((1, 2, 16, 16), -10.0)
    wall_masks[0, 0, 2:14, 4:6] = 10.0
    wall_masks[0, 1, 4:6, 2:14] = 10.0
    endpoints = wall_endpoints_from_masks(wall_masks)
    assert endpoints.shape == (1, 2, 4)
    assert (endpoints[..., 2:] >= endpoints[..., :2]).all()
    head = DynamicWallOpeningIntervalHead(16, 3)
    output = head(
        torch.randn(1, 2, 16),
        torch.randn(1, 3, 16),
        wall_masks,
    )
    assert output["presence_logits"].shape == (1, 2)
    assert output["type_logits"].shape == (1, 2, 3)
    assert output["center_xy"].shape == (1, 2, 2)
