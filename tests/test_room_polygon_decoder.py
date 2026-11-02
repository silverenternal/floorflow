from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.polygon_loop import polygon_loop_loss
from floorflow.heads.room_polygon_decoder import (
    RoomAwarePolygonHead,
    decode_polygons_with_dense_classes,
)


def test_room_aware_polygon_head_forward_backward() -> None:
    head = RoomAwarePolygonHead(
        16,
        3,
        hidden_dim=24,
        num_queries=6,
        points=10,
        layers=1,
        heads=4,
    )
    features = torch.randn(2, 16, 32, 32, requires_grad=True)
    room_logits = torch.randn(2, 3, 32, 32)
    predictions = head(features, room_logits)
    assert predictions["polygons"].shape == (2, 6, 10, 2)
    assert predictions["corner_count_logits"].shape == (2, 6, 8)
    assert predictions["query_hidden"].shape == (2, 6, 24)
    classes = torch.tensor([[0, 1, 0, 0, 1, 0], [2, 0, 0, 1, 0, 0]])
    polygons = torch.rand(2, 6, 10, 2)
    valid = torch.tensor(
        [
            [True, True, False, False, True, False],
            [True, True, False, False, False, False],
        ]
    )
    loss, components = polygon_loop_loss(
        predictions,
        classes,
        polygons,
        valid,
        mask_rasterizer="diff_ras",
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "room_polygon_loss",
        "room_polygon_class_loss",
        "room_polygon_corner_loss",
        "room_polygon_chamfer_loss",
        "room_polygon_mask_dice_loss",
        "room_polygon_valid_loss",
        "room_polygon_point_valid_loss",
    }
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert head.query_feature.weight.grad is not None


def test_polygon_decode_uses_dense_class_evidence() -> None:
    taxonomy = FloorplanTaxonomy(
        class_names=("background", "wall", "room:Bedroom", "room:Kitchen"),
        room_map={"Bedroom": "room:Bedroom", "Kitchen": "room:Kitchen"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={"room": {}, "furniture": {}, "door": {}, "window": {}},
    )
    polygons = torch.tensor(
        [[[[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]]]]
    )
    dense = torch.full((1, 2, 16, 16), -10.0)
    dense[:, 1, 4:12, 4:12] = 10.0
    decoded = decode_polygons_with_dense_classes(
        {
            "polygons": polygons,
            "valid_logits": torch.tensor([[2.0]]),
            "dense_logits": dense,
        },
        taxonomy,
        height=16,
        width=16,
    )
    assert decoded[0, 8, 8] == taxonomy.class_id("room:Kitchen")
