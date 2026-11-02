from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.training.cad_furniture_assignment import (
    furniture_room_assignment_loss,
    room_assignment_targets,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 5, "LivingRoom": 5}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_room_assignment_targets_use_surrounding_room() -> None:
    taxonomy = _taxonomy()
    labels = torch.full(
        (1, 16, 16),
        taxonomy.room_ids[0],
        dtype=torch.long,
    )
    labels[:, :, 8:] = taxonomy.room_ids[1]
    masks = torch.zeros(1, 3, 16, 16)
    masks[0, 0, 3:5, 3:5] = 1.0
    masks[0, 1, 3:5, 11:13] = 1.0
    valid = torch.tensor([[True, True, False]])
    targets = room_assignment_targets(
        torch.zeros(1, 3, dtype=torch.long),
        masks,
        valid,
        labels,
        taxonomy,
    )
    assert targets.tolist() == [[0, 1, 2]]


def test_assignment_loss_is_live_for_matched_and_background_queries() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    masks = torch.zeros(1, 1, 16, 16)
    masks[0, 0, 3:6, 3:6] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    class_logits = torch.full((1, 2, len(taxonomy.furniture_ids) + 1), -4.0)
    class_logits[0, 0, 0] = 4.0
    mask_logits = torch.full((1, 2, 16, 16), -4.0)
    mask_logits[0, 0, 3:6, 3:6] = 4.0
    assignment_logits = torch.zeros(
        1,
        2,
        len(taxonomy.room_ids) + 1,
        requires_grad=True,
    )
    loss, components = furniture_room_assignment_loss(
        {
            "class_logits": class_logits,
            "mask_logits": mask_logits,
            "room_assignment_logits": assignment_logits,
        },
        torch.zeros(1, 1, dtype=torch.long),
        masks,
        valid,
        labels,
        taxonomy,
    )
    loss.backward()
    assert assignment_logits.grad is not None
    assert assignment_logits.grad.abs().sum() > 0.0
    assert "furniture_room_assignment_accuracy" in components
