from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.recursion.inward_flow import (
    contour_inward_location,
    contour_inward_weight_map,
    inside_distance,
    multi_stage_weights,
    stage_family_weights,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_contour_band_moves_from_boundary_to_interior() -> None:
    footprint = torch.zeros((1, 32, 32))
    footprint[:, 2:30, 2:30] = 1
    distance = inside_distance(footprint)
    boundary_weight = contour_inward_weight_map(footprint, 0.0)
    center_weight = contour_inward_weight_map(footprint, 1.0)
    assert distance[0, 16, 16] > distance[0, 2, 16]
    assert boundary_weight[0, 2, 16] > boundary_weight[0, 16, 16]
    assert center_weight[0, 16, 16] > center_weight[0, 2, 16]

    source = torch.randn((1, 3, 32, 32), requires_grad=True)
    target = torch.randn((1, 3, 32, 32))
    location, velocity = contour_inward_location(
        source,
        target,
        footprint,
        0.5,
    )
    assert location.shape == source.shape
    assert velocity.shape == target.shape
    location.sum().backward()
    assert source.grad is not None


def test_multistage_weight_centers() -> None:
    early = multi_stage_weights(0.0)
    middle = multi_stage_weights(0.5)
    late = multi_stage_weights(1.0)
    assert early.room > early.structure and early.room > early.detail
    assert middle.structure > middle.room and middle.structure > middle.detail
    assert late.detail > late.room and late.detail > late.structure

    taxonomy = _taxonomy()
    labels = torch.zeros((1, 8, 8), dtype=torch.long)
    labels[:, 1:4, 1:4] = taxonomy.class_id("room:Bedroom")
    labels[:, 4, 1:4] = taxonomy.class_id("wall")
    labels[:, 5, 1:4] = taxonomy.class_id("door:Swing Beside")
    weights = stage_family_weights(labels, taxonomy, middle)
    assert weights[labels == taxonomy.class_id("room:Bedroom")].mean() > 0
    assert weights[labels == taxonomy.class_id("wall")].mean() > 0
    assert weights[labels == taxonomy.class_id("door:Swing Beside")].mean() > 0
