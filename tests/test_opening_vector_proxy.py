from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.opening_vector_proxy import opening_vector_proxy
from floorflow.geometry.opening_readout import (
    consolidate_opening_components,
)


def test_opening_vector_proxy_is_deterministic_for_identical_inputs() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[2:5, 2:30] = taxonomy.class_id("wall")
    labels[26:29, 2:30] = taxonomy.class_id("wall")
    labels[2:29, 2:5] = taxonomy.class_id("wall")
    labels[2:29, 27:30] = taxonomy.class_id("wall")
    labels[3:28, 3:27] = taxonomy.room_ids[0]
    labels[13:16, 26:29] = taxonomy.opening_ids[0]
    metrics = opening_vector_proxy(labels, labels, taxonomy)
    assert set(metrics) == {
        "opening_vector_precision_proxy",
        "opening_vector_recall_proxy",
        "opening_vector_f1_proxy",
        "opening_vector_predicted_count_proxy",
        "opening_vector_target_count_proxy",
        "opening_vector_geometry_matched_proxy",
        "opening_vector_family_consistent_rate_proxy",
        "opening_vector_center_error_proxy",
        "opening_vector_width_ratio_proxy",
        "opening_vector_door_to_window_proxy",
        "opening_vector_window_to_door_proxy",
        "opening_vector_predicted_doors_proxy",
        "opening_vector_predicted_windows_proxy",
        "opening_vector_target_doors_proxy",
        "opening_vector_target_windows_proxy",
        "opening_vector_matched_doors_proxy",
        "opening_vector_matched_windows_proxy",
        "opening_vector_unmatched_predicted_proxy",
        "opening_vector_unmatched_target_proxy",
    }
    assert 0.0 <= metrics["opening_vector_f1_proxy"] <= 1.0
    assert metrics["opening_vector_family_consistent_rate_proxy"] == 1.0


def test_opening_vector_proxy_separates_geometry_from_family_errors() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = np.zeros((32, 32), dtype=np.int64)
    target[2:5, 2:30] = taxonomy.class_id("wall")
    target[26:29, 2:30] = taxonomy.class_id("wall")
    target[2:29, 2:5] = taxonomy.class_id("wall")
    target[2:29, 27:30] = taxonomy.class_id("wall")
    target[3:28, 3:27] = taxonomy.room_ids[0]
    target[13:16, 26:29] = taxonomy.opening_ids[0]
    predicted = target.copy()
    predicted[target == taxonomy.opening_ids[0]] = taxonomy.opening_ids[1]
    metrics = opening_vector_proxy(predicted, target, taxonomy)
    assert metrics["opening_vector_f1_proxy"] == 0.0
    assert metrics["opening_vector_geometry_matched_proxy"] == 1.0
    assert metrics["opening_vector_family_consistent_rate_proxy"] == 0.0


def test_learned_family_corrects_opening_component_type() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = np.zeros((32, 32), dtype=np.int64)
    target[2:5, 2:30] = taxonomy.class_id("wall")
    target[26:29, 2:30] = taxonomy.class_id("wall")
    target[2:29, 2:5] = taxonomy.class_id("wall")
    target[2:29, 27:30] = taxonomy.class_id("wall")
    target[3:28, 3:27] = taxonomy.room_ids[0]
    door = taxonomy.opening_ids[0]
    window = taxonomy.opening_ids[1]
    target[13:16, 26:29] = door
    predicted = target.copy()
    predicted[target == door] = window
    family = np.zeros((2, 32, 32), dtype=np.float32)
    family[0] = 0.9
    family[1] = 0.1
    metrics = opening_vector_proxy(
        predicted,
        target,
        taxonomy,
        opening_family_probability=family,
    )
    assert metrics["opening_vector_f1_proxy"] == 1.0
    assert metrics["opening_vector_family_consistent_rate_proxy"] == 1.0


def test_opening_components_are_retyped_without_splitting() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing": 2}),
            "window": Counter({"Regular": 2}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((8, 8), dtype=np.int64)
    door, window = taxonomy.opening_ids[:2]
    labels[2:5, 2:6] = door
    labels[2:5, 4:7] = window
    family = np.zeros((2, 8, 8), dtype=np.float32)
    family[0] = 0.9
    family[1] = 0.1
    consolidated = consolidate_opening_components(
        labels,
        taxonomy,
        family,
    )
    component = np.isin(consolidated, [door, window])
    assert (consolidated[component] == door).all()
