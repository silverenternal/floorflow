from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.hybrid_decode import hybrid_decode
from floorflow.inference.room_repair import (
    apply_floorplan_repairs,
    close_room_partition,
    fill_room_support,
    merge_room_fragments,
)
from floorflow.objectives.semantic_losses import boundary_focal_dice
from floorflow.representation.core import (
    coverage_density,
    energy_distance,
    extract_room_instances,
    floorplan_feature_vector,
    hungarian_room_iou,
    mmd_rbf,
    room_class_adjacency,
)


def test_room_instances_and_hungarian_matching() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = np.zeros((32, 32), dtype=np.uint8)
    bedroom = taxonomy.class_id("room:Bedroom")
    target[2:14, 2:14] = bedroom
    target[18:30, 18:30] = bedroom
    prediction = target.copy()
    instances = extract_room_instances(target, taxonomy)
    assert len(instances) == 2
    scores = hungarian_room_iou(prediction, target, taxonomy)
    assert scores["matched_iou"] == 1.0
    assert scores["instance_precision"] == 1.0
    features = floorplan_feature_vector(target, taxonomy)[None]
    assert np.isfinite(mmd_rbf(features, features))
    assert abs(energy_distance(features, features)) < 1.0e-8
    assert coverage_density(features, features) == (1.0, 1.0)
    adjacency = room_class_adjacency(target, taxonomy)
    assert adjacency.shape == (0,)


def test_boundary_loss_and_room_repair() -> None:
    logits = torch.randn(1, 19, 16, 16)
    target = torch.zeros(1, 19, 16, 16)
    target[:, 0, 2:14, 2:14] = 1.0
    assert torch.isfinite(boundary_focal_dice(logits, target))
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[2:6, 2:6] = taxonomy.class_id("room:Bedroom")
    footprint = np.ones((16, 16), dtype=np.uint8)
    repaired = close_room_partition(merge_room_fragments(labels, taxonomy), footprint, taxonomy)
    assert repaired.shape == labels.shape
    secondary = labels.copy()
    secondary[6:10, 6:10] = 1
    assert hybrid_decode(labels, secondary, taxonomy).shape == labels.shape
    assert apply_floorplan_repairs(labels, footprint, taxonomy).shape == labels.shape
    wall_id = taxonomy.class_id("wall")
    labels[0, 0] = wall_id
    filled = fill_room_support(labels, footprint, taxonomy)
    assert filled[0, 0] == wall_id
    assert int((filled == 0).sum()) == 0
    support = np.zeros_like(footprint, dtype=bool)
    support[2:8, 2:8] = True
    seed = np.zeros_like(labels)
    seed[2:4, 2:4] = taxonomy.class_id("room:Bedroom")
    bounded = fill_room_support(
        seed,
        footprint,
        taxonomy,
        support_mask=support,
    )
    assert np.all(bounded[~support] == 0)
    assert np.all(bounded[support] != 0)
