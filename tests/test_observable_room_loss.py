from collections import Counter

import torch

from floorflow.objectives.observable_room_loss import (
    observable_room_loss,
    visible_room_semantics,
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


def test_visible_semantics_ignore_occluded_logits() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, :, :8] = taxonomy.room_ids[0]
    labels[:, :, 8:12] = taxonomy.room_ids[1]
    labels[:, :, 12:] = taxonomy.class_id("furniture:Sink")
    footprint = torch.ones(1, 16, 16)
    logits = torch.randn(1, 2, 16, 16)
    changed = logits.clone()
    changed[:, :, :, 12:] += 20.0
    first, _ = visible_room_semantics(logits, labels, footprint, taxonomy)
    second, _ = visible_room_semantics(changed, labels, footprint, taxonomy)
    assert torch.allclose(first, second)


def test_observable_loss_separates_support_from_semantics() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, :, :7] = taxonomy.room_ids[0]
    labels[:, :, 7:14] = taxonomy.room_ids[1]
    labels[:, :, 14:] = taxonomy.class_id("furniture:Sink")
    support = torch.zeros(1, 16, 16, dtype=torch.float32)
    support[:, :, :] = 1.0
    logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    loss, components = observable_room_loss(
        logits,
        labels,
        support,
        torch.ones(1, 16, 16),
        taxonomy,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert components["visible_room_pixels"].item() == 224
    assert components["room_support_pixels"].item() == 256
