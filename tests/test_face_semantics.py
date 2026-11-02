from collections import Counter

import numpy as np
import torch

from floorflow.heads.face_semantics import (
    FaceSemanticHead,
    build_face_targets,
    face_semantic_loss,
    face_semantic_metrics,
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
    labels[12:52, 12:30] = taxonomy.room_ids[0]
    labels[12:52, 34:52] = taxonomy.room_ids[1]
    labels[8:56, 8:12] = wall
    labels[8:56, 52:56] = wall
    labels[8:12, 8:56] = wall
    labels[52:56, 8:56] = wall
    labels[8:56, 30:34] = wall
    return taxonomy, labels


def test_face_targets_and_adjacency() -> None:
    taxonomy, labels = _fixture()
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    targets = build_face_targets(plan, labels, taxonomy)
    assert len(targets.masks) == 2
    assert targets.adjacency.shape == (2, 2)
    assert targets.adjacency[0, 1] > 0
    assert set(targets.classes) == set(range(len(taxonomy.room_ids)))


def test_face_semantic_head_loss_and_metrics() -> None:
    taxonomy, labels = _fixture()
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    targets = [build_face_targets(plan, labels, taxonomy)]
    head = FaceSemanticHead(12, len(taxonomy.room_ids), hidden_dim=32)
    predictions = head(torch.randn(1, 12, 64, 64), [targets[0].masks], [targets[0].adjacency])
    loss, components = face_semantic_loss(predictions, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.classifier.weight.grad is not None
    assert 0.0 <= face_semantic_metrics(predictions, targets)["face_semantic_accuracy"] <= 1.0
    assert "face_semantic_ce" in components


def test_face_semantic_head_handles_no_faces() -> None:
    head = FaceSemanticHead(4, 2, hidden_dim=8)
    predictions = head(
        torch.randn(1, 4, 16, 16),
        [[]],
        [np.eye(0, dtype=np.float32)],
    )
    loss, _ = face_semantic_loss(predictions, [])
    assert loss.item() == 0.0
