import numpy as np

from floorflow.inference.repair_features import repair_decision_features, target_free_repair_features
from floorflow.data.taxonomy import build_taxonomy
from collections import Counter


def test_repair_features_are_bounded_and_target_free() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[2:14, 2:14] = taxonomy.class_id("room:Bedroom")
    labels[2, 2:8] = taxonomy.class_id("wall")
    footprint = np.zeros((16, 16), dtype=np.uint8)
    footprint[1:15, 1:15] = 1
    features = target_free_repair_features(labels, footprint, taxonomy)
    assert 0.0 <= features["feature_room_coverage"] <= 1.0
    assert 0.0 <= features["feature_opening_alignment"] <= 1.0
    assert 0.0 <= features["feature_containment"] <= 1.0
    assert not any("iou" in name or "mae" in name for name in features)
    decision = repair_decision_features(labels, labels, footprint, taxonomy)
    assert "feature_raw_room_coverage" in decision
    assert "feature_candidate_room_coverage" in decision
    assert decision["feature_delta_room_coverage"] == 0.0
