from collections import Counter

import numpy as np
import torch

from floorflow.heads.geometry_heads import (
    HEATMAP_KEYS,
    GeometryHeatmapHeads,
    build_geometry_targets,
    geometry_heatmap_loss,
)
from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4, "Kitchen": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _fixture():
    taxonomy = _taxonomy()
    labels = np.zeros((64, 64), dtype=np.int64)
    wall = taxonomy.class_id("wall")
    labels[8:56, 30:34] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[12:52, 12:30] = taxonomy.room_ids[0]
    labels[12:52, 34:52] = taxonomy.room_ids[1]
    labels[28:36, 52:56] = taxonomy.opening_ids[0]
    return taxonomy, labels


def test_geometry_targets_are_peaked_and_loss_backpropagates() -> None:
    taxonomy, labels = _fixture()
    targets = build_geometry_targets(
        torch.from_numpy(labels)[None],
        torch.ones(1, 64, 64),
        taxonomy,
    )
    assert set(targets) == set(HEATMAP_KEYS)
    for target in targets.values():
        assert target.shape == (1, 1, 64, 64)
        assert float(target.max()) > 0.9
    heads = GeometryHeatmapHeads(16, hidden_dim=16)
    features = torch.randn(1, 16, 64, 64, requires_grad=True)
    predictions = heads(features)
    loss, components = geometry_heatmap_loss(predictions, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in heads.parameters()
    )
    assert set(components) == {
        "geometry_total",
        "geometry_junction",
        "geometry_wall_centerline",
        "geometry_opening_center",
        "geometry_room_boundary",
    }


def test_gt_centerline_heatmap_reconstructs_room_faces() -> None:
    taxonomy, labels = _fixture()
    targets = build_geometry_targets(
        torch.from_numpy(labels)[None],
        torch.ones(1, 64, 64),
        taxonomy,
    )
    wallmask = targets["wall_centerline"][0, 0].numpy() > 0.05
    reconstructed = np.zeros_like(labels)
    reconstructed[wallmask] = taxonomy.class_id("wall")
    plan = raster_labels_to_wall_plan(reconstructed, taxonomy)
    assert len(plan.walls) >= 4
    assert len(plan.rooms) >= 2


def test_geometry_targets_accept_room_without_wall() -> None:
    taxonomy, labels = _fixture()
    labels[labels == taxonomy.class_id("wall")] = taxonomy.room_ids[0]
    labels[np.isin(labels, taxonomy.opening_ids)] = taxonomy.room_ids[0]
    targets = build_geometry_targets(
        torch.from_numpy(labels)[None],
        torch.ones(1, 64, 64),
        taxonomy,
    )
    assert float(targets["wall_centerline"].sum()) == 0.0
