from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.control_baseline import control_layers_batch


def test_control_layers_batch_preserves_sample_geometry() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room_count = len(taxonomy.room_ids)
    controls = {
        "room_logits": torch.zeros(2, room_count, 8, 8),
        "program": torch.zeros(2, room_count, dtype=torch.bool),
        "structure": torch.zeros(
            2,
            room_count + len(taxonomy.opening_ids) + 3,
            8,
            8,
        ),
        "furniture": torch.zeros(
            2,
            room_count + len(taxonomy.furniture_ids) + 1,
            8,
            8,
        ),
        "support": torch.zeros(2, room_count + 4, 8, 8),
    }
    controls["room_logits"][0, 0, :4] = 2.0
    controls["room_logits"][1, 1, 4:] = 2.0
    footprint = torch.ones(2, 1, 16, 16)
    rooms, wall, openings, furniture = control_layers_batch(
        controls,
        footprint,
        taxonomy,
    )
    assert rooms.shape == (2, 16, 16)
    assert wall.shape == (2, 16, 16)
    assert openings.shape == (2, 16, 16)
    assert furniture.shape == (2, 16, 16)
    assert rooms[0, 0, 0].item() == taxonomy.room_ids[0]
    assert rooms[1, 15, 0].item() == taxonomy.room_ids[1]
