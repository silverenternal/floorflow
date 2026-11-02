from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.dense_colorized_raster2seq import (
    dense_colorized_raster2seq_input,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "room:LivingRoom",
            "room:Kitchen",
            "door:Swing Beside",
            "furniture:Chair",
        ),
        room_map={"LivingRoom": "room:LivingRoom", "Kitchen": "room:Kitchen"},
        furniture_map={"Chair": "furniture:Chair"},
        door_map={"Swing Beside": "door:Swing Beside"},
        window_map={},
        source_counts={
            "room": {},
            "furniture": {"Chair": 1},
            "door": {},
            "window": {},
        },
    )


def test_rooms_mode_uses_only_target_free_room_logits() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones(1, 16, 16)
    room_logits = torch.zeros(1, 2, 16, 16)
    room_logits[:, 1, 4:12, 4:12] = 10.0
    result = dense_colorized_raster2seq_input(
        {"room_logits": room_logits},
        footprint,
        taxonomy,
        mode="rooms",
    )
    assert result.image.shape == (16, 16, 3)
    assert result.wall_pixels == 0
    assert result.opening_pixels == 0
    assert (result.labels[4:12, 4:12] == taxonomy.class_id("room:Kitchen")).all()


def test_full_structure_mode_uses_predicted_wall_support() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones(1, 16, 16)
    room_logits = torch.zeros(1, 2, 16, 16)
    room_logits[:, 0] = 10.0
    wall_logits = torch.full((1, 16, 16), -10.0)
    wall_logits[0, :, 7:9] = 10.0
    opening_logits = torch.full((1, 1, 16, 16), -10.0)
    furniture_logits = torch.full((1, 1, 16, 16), -10.0)
    result = dense_colorized_raster2seq_input(
        {
            "room_logits": room_logits,
            "wall_logits": wall_logits,
            "opening_logits": opening_logits,
            "furniture_logits": furniture_logits,
        },
        footprint,
        taxonomy,
        mode="full_structure",
        wall_probability_threshold=0.5,
    )
    assert result.wall_pixels == 32
    assert (result.labels[:, 7] == taxonomy.class_id("wall")).all()
