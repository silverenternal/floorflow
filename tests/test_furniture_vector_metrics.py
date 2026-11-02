from collections import Counter

import cv2
import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.furniture_vector_metrics import furniture_vector_metrics
from floorflow.geometry.wall_graph_readout import ReadoutFurniture


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_furniture_vector_metrics_detect_attachment_and_containment() -> None:
    taxonomy = _taxonomy()
    room_id = taxonomy.room_ids[0]
    furniture_id = taxonomy.furniture_ids[0]
    target = np.full((16, 16), room_id, dtype=np.int32)
    polygon = ((4, 4), (9, 4), (9, 8), (4, 8))
    cv2.fillPoly(
        target,
        [np.asarray(polygon, dtype=np.int32)],
        int(furniture_id),
    )
    target = target.astype(np.int64)
    rooms = [
        {
            "label": taxonomy.class_names[room_id],
            "polygon": [[0, 0], [16, 0], [16, 16], [0, 16]],
        }
    ]
    furniture = [
        ReadoutFurniture(
            label=taxonomy.class_names[furniture_id],
            polygon=polygon,
            room_index=0,
        )
    ]
    metrics = furniture_vector_metrics(
        furniture,
        rooms,
        target,
        taxonomy,
    )
    assert metrics["furniture_vector_f1"] == 1.0
    assert metrics["furniture_vector_attachment_accuracy"] == 1.0
    assert metrics["furniture_vector_matched_label_accuracy"] == 1.0
    assert metrics["furniture_vector_matched_target_iou"] == 1.0
    assert metrics["furniture_vector_containment"] == 1.0
    assert metrics["furniture_vector_polygon_valid_rate"] == 1.0
    assert metrics["furniture_vector_out_of_boundary"] == 0.0
