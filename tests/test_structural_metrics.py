from __future__ import annotations

import numpy as np
import torch
from collections import Counter

from floorflow.evaluation.structural_metrics import connected_component_count, structural_metrics
from floorflow.evaluation.wall_metrics import thin_binary_mask
from floorflow.evaluation.wall_metrics import wall_fields
from floorflow.geometry.interior_wall import InteriorWallHead, interior_wall_iou, interior_wall_loss
from floorflow.objectives.design_reward import structural_reward_from_labels
from floorflow.data.taxonomy import build_taxonomy


def test_connected_component_count() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    assert connected_component_count(mask) == 2


def test_structural_reward_shape() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 2:14, 2:14] = taxonomy.class_id("room:Bedroom")
    footprint = torch.ones(1, 16, 16)
    reward = structural_reward_from_labels(labels, labels, footprint, taxonomy)
    assert reward.shape == (4,)


def test_ground_truth_furniture_is_fully_contained() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter({"Bed": 4}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 2:14, 2:14] = taxonomy.class_id("room:Bedroom")
    labels[:, 6:10, 5:11] = taxonomy.class_id("furniture:Bed")
    footprint = torch.ones(1, 16, 16)
    metrics = structural_metrics(labels, labels, footprint, taxonomy)
    assert metrics["furniture_room_containment"] == 1.0


def test_empty_furniture_prediction_has_vacuous_containment() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter({"Bed": 4}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 2:14, 2:14] = taxonomy.class_id("room:Bedroom")
    metrics = structural_metrics(labels, labels, torch.ones(1, 16, 16), taxonomy)
    assert metrics["furniture_room_containment"] == 1.0


def test_wall_thinning_preserves_a_line() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[7:10, 2:14] = True
    thinned = thin_binary_mask(mask)
    assert thinned.sum() < mask.sum()
    assert thinned[:, 8].any()


def test_wall_openings_form_a_closed_structure() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, 4:6, 4:28] = taxonomy.class_id("wall")
    labels[:, 26:28, 4:28] = taxonomy.class_id("wall")
    labels[:, 4:28, 4:6] = taxonomy.class_id("wall")
    labels[:, 4:28, 26:28] = taxonomy.class_id("wall")
    labels[:, 4:6, 15:17] = taxonomy.opening_ids[0]
    metrics = wall_fields(labels, labels, taxonomy)
    assert metrics["wall_closure"] > 0.9


def test_interior_wall_head_and_loss() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    head = InteriorWallHead(16)
    features = torch.randn(1, 16, 32, 32)
    logits = head(features)
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 15:17, 6:26] = taxonomy.class_id("wall")
    footprint = torch.ones(1, 32, 32)
    loss, _, _ = interior_wall_loss(logits, target, footprint, taxonomy)
    assert torch.isfinite(loss)
    assert 0.0 <= interior_wall_iou(logits, target, footprint, taxonomy) <= 1.0
