from collections import Counter

import torch

from floorflow.heads.room_program import (
    RoomProgramHead,
    cardinality_gated_program,
    room_presence_target,
    room_program_loss,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10, "Bath": 10}),
            "furniture": Counter({"Toilet": 10}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_room_program_presence_and_bias_backpropagate() -> None:
    taxonomy = _taxonomy()
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[:, :8] = taxonomy.class_id("room:Bedroom")
    labels[:, 8:] = taxonomy.class_id("room:Kitchen")
    head = RoomProgramHead(12, len(taxonomy.room_ids), hidden_dim=32, tokens=4)
    predictions = head(torch.randn(2, 12, 16, 16))
    assert predictions["presence_logits"].shape == (2, len(taxonomy.room_ids))
    target = room_presence_target(labels, taxonomy)
    assert target.shape == predictions["presence_logits"].shape
    loss, components = room_program_loss(predictions, labels, taxonomy)
    assert torch.isfinite(loss)
    assert "room_program_loss" in components
    loss.backward()
    assert head.presence.weight.grad is not None


def test_cardinality_gated_program_uses_predicted_probability_mass() -> None:
    logits = torch.tensor([[-4.0, 2.0, 2.0, -4.0, 1.0]])
    selected = cardinality_gated_program(logits)
    assert selected.tolist() == [[False, True, True, False, True]]
