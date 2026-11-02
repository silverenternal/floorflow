from collections import Counter

import numpy as np

from floorflow.evaluation.conditional_ceiling import (
    distribution_entropy,
    neighbour_iou_summary,
    room_macro_iou_numpy,
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


def test_macro_iou_and_neighbour_summary() -> None:
    taxonomy = _taxonomy()
    bedroom, kitchen = taxonomy.room_ids
    target = np.asarray([[bedroom, bedroom], [kitchen, kitchen]], dtype=np.uint8)
    perfect = target.copy()
    empty = np.zeros_like(target)
    assert room_macro_iou_numpy(perfect, target, taxonomy) == 1.0
    summary = neighbour_iou_summary(
        np.stack([perfect, empty]),
        target,
        taxonomy,
    )
    assert summary["top1_iou"] == 1.0
    assert summary["neighbour_mean_iou"] == 0.5
    assert summary["neighbour_max_iou"] == 1.0


def test_distribution_entropy_is_zero_for_single_mode() -> None:
    assert distribution_entropy(np.asarray([1.0, 0.0])) == 0.0
    assert distribution_entropy(np.asarray([0.5, 0.5])) > 0.0
