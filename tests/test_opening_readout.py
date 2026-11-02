from collections import Counter

import numpy as np

from floorflow.geometry.opening_readout import (
    detections_from_labels,
    opening_detection_metrics,
    read_opening_detections,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter({"Regular": 4}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_detector_readout_matches_family_aware_centres() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((32, 32), dtype=np.int64)
    labels[10:20, 12:16] = taxonomy.class_id("door:Swing Beside")
    labels[22:26, 20:26] = taxonomy.class_id("window:Regular")
    target = detections_from_labels(labels, taxonomy)
    probabilities = np.stack(
        [
            (labels == taxonomy.class_id("door:Swing Beside")).astype(np.float32),
            (labels == taxonomy.class_id("window:Regular")).astype(np.float32),
        ]
    )
    predicted = read_opening_detections(
        probabilities,
        taxonomy.opening_ids,
        taxonomy.class_names,
        taxonomy,
        threshold=0.5,
    )
    metrics = opening_detection_metrics(predicted, target, tolerance=2.0)
    assert metrics["f1"] == 1.0
    assert metrics["matched"] == 2
