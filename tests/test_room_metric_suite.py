from collections import Counter

import numpy as np

from floorflow.evaluation.room_metric_suite import (
    matched_instance_metrics,
    room_metric_suite,
)
from floorflow.data.taxonomy import build_taxonomy


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


def _rooms(taxonomy):
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[4:28, 4:16] = taxonomy.room_ids[0]
    labels[4:28, 16:28] = taxonomy.room_ids[1]
    return labels


def test_perfect_rooms_have_three_aligned_metrics() -> None:
    taxonomy = _taxonomy()
    labels = _rooms(taxonomy)
    footprint = np.ones_like(labels)
    report = room_metric_suite(
        labels,
        labels,
        footprint,
        taxonomy,
        predicted_room_support=labels > 0,
        target_room_support=labels > 0,
    )
    assert report["visible_semantic"]["room_macro_iou"] == 1.0
    assert report["occlusion_aware_support"]["room_support_iou"] == 1.0
    assert report["support_instance"]["matched_geometric_iou"] == 1.0
    assert report["matched_instance"]["instance_f1_at_threshold"] == 1.0
    assert report["matched_instance"]["matched_type_accuracy"] == 1.0


def test_swapped_types_separate_geometry_from_semantics() -> None:
    taxonomy = _taxonomy()
    target = _rooms(taxonomy)
    prediction = target.copy()
    left = target == taxonomy.room_ids[0]
    right = target == taxonomy.room_ids[1]
    prediction[left] = taxonomy.room_ids[1]
    prediction[right] = taxonomy.room_ids[0]
    report = room_metric_suite(
        prediction,
        target,
        np.ones_like(target),
        taxonomy,
    )
    assert report["visible_semantic"]["room_macro_iou"] == 0.0
    assert report["matched_instance"]["instance_f1_at_threshold"] == 1.0
    assert report["matched_instance"]["matched_type_accuracy"] == 0.0
    assert report["matched_instance"]["instance_class_constrained_f1_at_threshold"] == 0.0
    assert report["matched_instance"]["matched_class_constrained_iou"] == 0.0
    assert report["occlusion_aware_support"]["available"] is False
    assert report["support_instance"] is None


def test_instance_metrics_detect_count_and_mask_errors() -> None:
    taxonomy = _taxonomy()
    target = _rooms(taxonomy)
    prediction = target.copy()
    prediction[:, 16:] = 0
    metrics = matched_instance_metrics(prediction, target, taxonomy)
    assert metrics["predicted_instances"] == 1
    assert metrics["target_instances"] == 2
    assert metrics["instance_recall_at_threshold"] == 0.5
