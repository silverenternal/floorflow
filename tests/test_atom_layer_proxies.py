from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.atom_layer_proxies import layer_proxy_metrics


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Door": 2}),
            "window": Counter({"Window": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_layer_proxy_metrics_are_bounded_and_sensitive() -> None:
    taxonomy = _taxonomy()
    target = np.zeros((32, 32), dtype=np.uint8)
    predicted = target.copy()
    room = taxonomy.class_id("room:Bedroom")
    wall = taxonomy.class_id("wall")
    door = taxonomy.class_id("door:Door")
    sink = taxonomy.class_id("furniture:Sink")
    target[4:28, 4:28] = room
    predicted[4:28, 4:28] = room
    target[4:28, 14:16] = wall
    predicted[4:28, 14:16] = wall
    target[6:10, 6:10] = sink
    predicted[6:10, 6:10] = sink
    target[15:17, 14:16] = door
    predicted[15:17, 14:16] = door
    metrics = layer_proxy_metrics(
        predicted,
        target,
        np.ones((32, 32), dtype=bool),
        taxonomy,
        predicted_room_support=target != wall,
    )
    assert metrics["wall_centerline_f1_proxy"] == 1.0
    assert metrics["opening_wall_alignment_proxy"] == 1.0
    assert metrics["furniture_predicted_room_containment_proxy"] == 1.0

    disrupted = predicted.copy()
    disrupted[6:10, 6:10] = 0
    worsened = layer_proxy_metrics(
        disrupted,
        target,
        np.ones((32, 32), dtype=bool),
        taxonomy,
        predicted_room_support=target != wall,
    )
    assert worsened["furniture_count_abs_error_proxy"] == 1
