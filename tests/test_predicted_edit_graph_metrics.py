from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.predicted_edit_graph_metrics import (
    predicted_edit_graph_metrics,
)


def test_predicted_edit_graph_metrics_separate_type_and_instance_scopes() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1, "Kitchen": 1}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[2:14, 2:8] = taxonomy.room_ids[0]
    labels[2:14, 8:14] = taxonomy.room_ids[1]
    footprint = labels > 0
    metrics = predicted_edit_graph_metrics(
        labels,
        labels,
        taxonomy,
        visible=labels,
        footprint=footprint,
        timeout_seconds=1.0,
    )
    assert metrics["refined"]["node_f1"] == 1.0
    assert metrics["refined"]["edge_overlap"] == 1.0
    assert metrics["refined"]["ged"] == 0.0
    assert metrics["refined"]["instance_node_f1"] == 1.0
    assert metrics["refined"]["instance_ged"] == 0.0
