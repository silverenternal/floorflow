from collections import Counter

import torch

from floorflow.objectives.structural_consistency import structural_consistency_loss
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


def test_structural_terms_have_visible_gradients() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, :7] = taxonomy.room_ids[0]
    target[:, :, 7:14] = taxonomy.room_ids[1]
    target[:, :, 14:16] = taxonomy.class_id("wall")
    target[:, 7:9, 14:16] = taxonomy.opening_ids[0]
    footprint = torch.ones(1, 16, 16)
    room_logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    objects = {
        "opening_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "opening_background_logits": torch.randn(1, 16, 16, requires_grad=True),
        "furniture_logits": torch.randn(1, 1, 16, 16, requires_grad=True),
        "furniture_background_logits": torch.randn(1, 16, 16, requires_grad=True),
    }
    loss, components = structural_consistency_loss(
        room_logits,
        objects,
        target,
        footprint,
        taxonomy,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert room_logits.grad is not None
    for tensor in objects.values():
        assert tensor.grad is not None
    assert set(components) == {
        "opening_wall_affinity",
        "room_wall_edge",
        "furniture_room_containment",
        "structural_consistency",
    }
