from __future__ import annotations

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.dense_room_wall_fusion import fuse_dense_room_walls
from floorflow.representation.core import extract_room_instances


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=("background", "wall", "room:Bedroom", "room:Bath"),
        room_map={"Bedroom": "room:Bedroom", "Bath": "room:Bath"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )


def test_predicted_wall_splits_same_class_room_component() -> None:
    taxonomy = _taxonomy()
    labels = np.full((16, 16), taxonomy.class_id("room:Bedroom"), dtype=np.int64)
    footprint = np.ones_like(labels, dtype=bool)
    wall_probability = np.zeros_like(labels, dtype=np.float32)
    wall_probability[:, 8] = 1.0
    baseline = extract_room_instances(labels, taxonomy, min_area=4)
    result = fuse_dense_room_walls(
        labels,
        footprint,
        wall_probability,
        taxonomy,
        mode="predicted_wall",
        wall_threshold=0.5,
    )
    split = extract_room_instances(result.labels, taxonomy, min_area=4)
    assert len(baseline) == 1
    assert len(split) == 2
    assert np.all(
        result.labels[:, 8] == taxonomy.class_id("wall")
    )
    assert not np.isin(result.labels[~footprint], taxonomy.room_ids).any()


def test_boundary_fusion_does_not_split_uniform_room() -> None:
    taxonomy = _taxonomy()
    labels = np.full((16, 16), taxonomy.class_id("room:Bedroom"), dtype=np.int64)
    result = fuse_dense_room_walls(
        labels,
        np.ones_like(labels, dtype=bool),
        np.zeros_like(labels, dtype=np.float32),
        taxonomy,
        mode="boundary_wall",
    )
    assert np.array_equal(result.labels, labels)


def test_wall_gated_mode_intersects_room_seams_with_wall_evidence() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[:, :4] = taxonomy.class_id("room:Bedroom")
    labels[:, 4:] = taxonomy.class_id("room:Bath")
    wall_probability = np.zeros((8, 8), dtype=np.float32)
    wall_probability[:, 3:5] = 0.9
    wall_probability[0, 3:5] = 0.1
    result = fuse_dense_room_walls(
        labels.numpy(),
        np.ones_like(labels.numpy(), dtype=bool),
        wall_probability,
        taxonomy,
        mode="wall_gated",
        wall_threshold=0.2,
    )
    wall = result.labels == taxonomy.class_id("wall")
    assert wall.any()
    assert not wall[0].any()


def test_owner_boundary_splits_same_class_rooms() -> None:
    taxonomy = _taxonomy()
    labels = np.full(
        (8, 8),
        taxonomy.class_id("room:Bedroom"),
        dtype=np.int64,
    )
    owner_boundary = np.zeros((8, 8), dtype=bool)
    owner_boundary[:, 4] = True
    result = fuse_dense_room_walls(
        labels,
        np.ones_like(labels, dtype=bool),
        np.zeros_like(labels, dtype=np.float32),
        taxonomy,
        mode="owner_boundary",
        owner_boundary=owner_boundary,
    )
    assert np.all(result.labels[:, 4] == taxonomy.class_id("wall"))
