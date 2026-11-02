"""Tests for channel-native zero-tolerance edit projection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from shapely.geometry import Polygon

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.vector_edit_protocol import (
    VectorOpening,
    VectorPlan,
    VectorRoom,
    VectorWall,
)
from floorflow.inference.strict_local_projection import (
    native_control_regions,
    native_region,
    outside_change_counts,
    project_controls,
    project_labels,
    project_vector_plan,
    typed_edge_exact,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 3, "Kitchen": 3}),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_native_region_does_not_expand_across_resolution() -> None:
    region = np.zeros((16, 16), dtype=bool)
    region[4:8, 4:8] = True
    resized = native_region(region, (16, 16), (8, 8))
    assert resized.shape == (8, 8)
    assert resized.sum() == 4
    assert not resized[:2].any()


def test_control_and_label_projection_restore_all_outside_cells() -> None:
    taxonomy = _taxonomy()
    rooms = torch.full((2, 16, 16), -8.0)
    rooms[0, :, :8] = 8.0
    rooms[1, :, 8:] = 8.0
    structure = torch.zeros(
        2 + len(taxonomy.opening_ids) + 3,
        16,
        16,
    )
    structure[2, :, 7:9] = 1.0
    furniture = torch.zeros(
        2 + len(taxonomy.furniture_ids) + 1,
        16,
        16,
    )
    support = torch.ones(2 + 4, 16, 16)
    before = {
        "room_logits": rooms,
        "program": torch.tensor([True, True]),
        "structure": structure,
        "furniture": furniture,
        "support": support,
    }
    after = {key: value.clone() for key, value in before.items()}
    after["room_logits"][0, 0, 0] = 9.0
    after["structure"][2, 0, 0] = 1.0
    region = np.zeros((16, 16), dtype=bool)
    region[:, 8:] = True
    projected = project_controls(
        before,
        after,
        native_control_regions(before, region),
    )
    counts = outside_change_counts(
        before,
        projected,
        native_control_regions(before, region),
    )
    assert all(value == 0 for value in counts.values())

    labels = torch.zeros(16, 16, dtype=torch.long)
    labels[:, :8] = taxonomy.room_ids[0]
    labels[:, 8:] = taxonomy.room_ids[1]
    attacked = labels.clone()
    attacked[0, 0] = taxonomy.room_ids[1]
    repaired = project_labels(
        labels.numpy(),
        attacked.numpy(),
        labels.numpy(),
        region,
        np.ones((16, 16), dtype=bool),
        taxonomy,
    )
    assert np.array_equal(repaired[~region], labels.numpy()[~region])


def test_outside_change_counts_accepts_cuda_tensors_when_available() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shape = (1, 4, 4)
    region = np.zeros((4, 4), dtype=bool)
    region[1:3, 1:3] = True
    before = {
        "room_logits": torch.zeros(shape, device=device),
        "structure": torch.zeros(shape, device=device),
        "furniture": torch.zeros(shape, device=device),
        "support": torch.zeros(shape, device=device),
    }
    after = {key: value.clone() for key, value in before.items()}
    after["room_logits"][0, 0, 0] = 1.0
    counts = outside_change_counts(
        before,
        after,
        {key: region for key in before},
    )
    assert counts["room_logits"] == 1


def test_unaffected_typed_edges_must_remain_equal() -> None:
    before = torch.eye(4, dtype=torch.bool)
    after = before.clone()
    after[0, 1] = after[1, 0] = True
    unaffected = torch.tensor([True, True, True, True])
    assert not typed_edge_exact(before, after, unaffected)
    unaffected[0] = False
    assert typed_edge_exact(before, after, unaffected)


def test_vector_projection_supports_room_deletion_without_zip_mismatch() -> None:
    left_polygon = Polygon([(0, 0), (8, 0), (8, 16), (0, 16)])
    right_polygon = Polygon([(8, 0), (16, 0), (16, 16), (8, 16)])
    left_mask = np.zeros((16, 16), dtype=bool)
    left_mask[:, :8] = True
    right_mask = np.zeros((16, 16), dtype=bool)
    right_mask[:, 8:] = True
    left = VectorRoom("Bedroom", left_polygon, left_mask)
    right = VectorRoom("Kitchen", right_polygon, right_mask)
    before = VectorPlan(
        rooms=(left, right),
        walls=(),
        openings=(),
        wall_mask=np.zeros((16, 16), dtype=bool),
        opening_mask=np.zeros((16, 16), dtype=bool),
        source=None,
    )
    after = VectorPlan(
        rooms=(left,),
        walls=(),
        openings=(),
        wall_mask=np.zeros((16, 16), dtype=bool),
        opening_mask=np.zeros((16, 16), dtype=bool),
        source=None,
    )
    region = np.zeros((16, 16), dtype=bool)
    region[:, 8:] = True
    projected = project_vector_plan(
        before,
        after,
        region,
        np.ones((16, 16), dtype=bool),
    )
    assert len(projected.rooms) == 1
    assert projected.rooms[0].polygon.equals(left_polygon)


def test_vector_projection_detects_one_pixel_wall_edit_region() -> None:
    wall = VectorWall(
        start=(0.0, 0.0),
        end=(16.0, 0.0),
        thickness=2.0,
        wall_type="interior",
    )
    empty_rooms = ()
    masks = np.zeros((16, 16), dtype=bool)
    before = VectorPlan(
        rooms=empty_rooms,
        walls=(wall,),
        openings=(),
        wall_mask=masks,
        opening_mask=masks,
        source=None,
    )
    after = VectorPlan(
        rooms=empty_rooms,
        walls=(),
        openings=(),
        wall_mask=masks,
        opening_mask=masks,
        source=None,
    )
    region = np.zeros((16, 16), dtype=bool)
    region[0, 5] = True
    projected = project_vector_plan(
        before,
        after,
        region,
        np.ones((16, 16), dtype=bool),
    )
    assert projected.walls == ()


def test_vector_projection_detects_opening_interval_outside_center() -> None:
    wall = VectorWall(
        start=(0.0, 8.0),
        end=(16.0, 8.0),
        thickness=2.0,
        wall_type="interior",
    )
    opening = VectorOpening(center=(3.0, 8.0), width=6.0, kind="door")
    masks = np.zeros((16, 16), dtype=bool)
    before = VectorPlan(
        rooms=(),
        walls=(wall,),
        openings=(opening,),
        wall_mask=masks,
        opening_mask=masks,
        source=None,
    )
    after = VectorPlan(
        rooms=(),
        walls=(wall,),
        openings=(),
        wall_mask=masks,
        opening_mask=masks,
        source=None,
    )
    region = np.zeros((16, 16), dtype=bool)
    region[8, 5] = True
    projected = project_vector_plan(
        before,
        after,
        region,
        np.ones((16, 16), dtype=bool),
    )
    assert projected.openings == ()
