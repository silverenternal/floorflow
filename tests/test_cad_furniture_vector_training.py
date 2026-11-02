from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.room_furniture_vector_head import (
    RoomFurnitureVectorHead,
)
from floorflow.training.cad_furniture_vector_training import (
    furniture_vector_target_loss,
    furniture_vector_targets,
)


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


def test_room_vector_head_is_differentiable_and_room_bounded() -> None:
    taxonomy = _taxonomy()
    head = RoomFurnitureVectorHead(
        8,
        len(taxonomy.room_ids),
        len(taxonomy.furniture_ids),
        hidden_dim=16,
        max_per_room=3,
        heads=4,
        layers=1,
    )
    features = torch.randn(2, 8, 16, 16, requires_grad=True)
    room_logits = torch.randn(2, len(taxonomy.room_ids), 16, 16)
    output = head(features, room_logits, render_grid=32)
    assert output["furniture_polygons"].shape == (
        2,
        len(taxonomy.room_ids),
        3,
        4,
        2,
    )
    assert output["furniture_polygons"].min() >= 0.0
    assert output["furniture_polygons"].max() <= 1.0
    assert output["furniture_render"].shape == (2, 1, 3, 32, 32)
    loss = output["furniture_class_logits"].sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_vector_targets_and_loss_use_direct_room_slots() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, :] = taxonomy.room_ids[0]
    target[:, 8:14, 8:15] = taxonomy.furniture_ids[0]
    targets = furniture_vector_targets(
        target,
        taxonomy,
        max_per_room=3,
    )
    assert targets["exists"].shape == (1, 1, 3)
    assert float(targets["exists"].sum()) == 1.0
    head = RoomFurnitureVectorHead(
        8,
        len(taxonomy.room_ids),
        len(taxonomy.furniture_ids),
        hidden_dim=16,
        max_per_room=3,
        heads=4,
        layers=1,
    )
    output = head(
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 1, 16, 16),
        render_grid=32,
    )
    loss, components = furniture_vector_target_loss(
        output,
        targets,
        render_grid=32,
        class_weights=torch.ones(len(taxonomy.furniture_ids)),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert components["vector_render_loss"] >= 0.0
