from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.inference.trinity_refine import repair_room_raster


def test_trinity_repair_returns_valid_labelled_rooms() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.zeros(1, 32, 32, dtype=torch.long)
    labels[:, :16] = taxonomy.room_ids[0]
    labels[:, 16:] = taxonomy.room_ids[1]
    footprint = torch.ones(1, 1, 32, 32)
    repaired = repair_room_raster(
        labels,
        footprint,
        taxonomy,
        steps=2,
        lr=1.0e-4,
    )
    assert repaired.shape == labels.shape
    assert int((repaired == taxonomy.room_ids[0]).sum()) > 0
    assert int((repaired == taxonomy.room_ids[1]).sum()) > 0
