from collections import Counter

import torch

from floorflow.evaluation.room_module_attribution import (
    labels_from_room_logits,
    room_support_metrics,
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


def test_room_support_attribution_metrics() -> None:
    taxonomy = _taxonomy()
    bedroom, kitchen = taxonomy.room_ids
    footprint = torch.ones(1, 4, 4)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[:, :, :2] = bedroom
    target[:, :, 2:] = kitchen
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 0] = 1.0
    prediction = labels_from_room_logits(logits, footprint, taxonomy)
    metrics = room_support_metrics(prediction, target, footprint, taxonomy)
    assert prediction.shape == target.shape
    assert 0.0 <= metrics["room_macro_iou"] <= 1.0
    assert 0.0 <= metrics["room_coverage"] <= 1.0
