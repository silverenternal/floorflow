from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.room_groups import (
    ROOM_GROUPS,
    group_room_logits,
    room_class_group_index,
    room_group_loss,
    room_group_macro_iou,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter(
                {
                    "Bedroom": 10,
                    "Bath": 10,
                    "Kitchen": 10,
                    "Hall": 10,
                    "Garage": 10,
                    "LivingRoom": 10,
                    "Dining": 10,
                    "Storage": 10,
                }
            ),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
    )


def test_group_logits_and_loss_are_hierarchical() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :4] = taxonomy.room_ids.index(taxonomy.class_id("room:Bedroom"))
    target[:, 4:8] = taxonomy.room_ids.index(taxonomy.class_id("room:Bath"))
    target[:, 8:12] = taxonomy.room_ids.index(taxonomy.class_id("room:Kitchen"))
    target[:, 12:] = taxonomy.room_ids.index(taxonomy.class_id("room:LivingRoom"))
    logits = torch.randn(1, len(taxonomy.room_ids), 16, 16, requires_grad=True)
    group_index = room_class_group_index(taxonomy)
    grouped = group_room_logits(logits, group_index)
    assert grouped.shape == (1, len(ROOM_GROUPS), 16, 16)
    loss, components = room_group_loss(logits, target, taxonomy)
    assert torch.isfinite(loss)
    assert 0.0 <= components["room_group_loss"] <= 100.0
    loss.backward()
    assert logits.grad is not None


def test_group_iou_merges_related_room_types() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = taxonomy.class_id("room:LivingRoom")
    prediction = target.clone()
    prediction[:, :4] = taxonomy.class_id("room:Dining")
    assert room_group_macro_iou(prediction, target, taxonomy) == 1.0


def test_group_loss_handles_invalid_target_without_host_branch() -> None:
    taxonomy = _taxonomy()
    target = torch.full((1, 8, 8), -1, dtype=torch.long)
    logits = torch.randn(
        1,
        len(taxonomy.room_ids),
        8,
        8,
        requires_grad=True,
    )
    loss, components = room_group_loss(logits, target, taxonomy)
    assert loss.item() == 0.0
    assert components["room_group_dice"].item() == 0.0
    loss.backward()
    assert logits.grad is not None
