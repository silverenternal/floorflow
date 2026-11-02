from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.control_contract import oracle_control_payload
from floorflow.data.taxonomy import FloorplanTaxonomy


def test_control_baseline_helpers_import() -> None:
    taxonomy = FloorplanTaxonomy.read(
        Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json")
    )
    from floorflow.evaluation.control_baseline import control_layers

    room_id = taxonomy.room_ids[0]
    labels = torch.full((1, 32, 32), room_id, dtype=torch.long)
    footprint = torch.ones(1, 32, 32)
    controls = oracle_control_payload(
        labels,
        footprint,
        taxonomy,
        room_size=8,
        structure_size=8,
        furniture_size=8,
        support_size=8,
    )
    rooms, wall, openings, furniture = control_layers(
        {key: value[0] for key, value in controls.items()},
        footprint[0],
        taxonomy,
    )
    assert rooms.shape == (32, 32)
    assert wall.shape == (32, 32)
    assert openings.shape == (32, 32)
    assert furniture.shape == (32, 32)
    assert int((rooms == room_id).sum()) > 0
