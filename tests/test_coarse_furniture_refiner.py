from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_furniture_refiner import (
    CoarseFurnitureRefiner,
    coarse_furniture_control,
)


def test_coarse_furniture_refiner_shapes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.full((1, 32, 32), taxonomy.room_ids[0])
    footprint = torch.ones(1, 32, 32)
    control = coarse_furniture_control(
        labels,
        footprint,
        taxonomy,
        size=8,
    )
    model = CoarseFurnitureRefiner(
        control.shape[1],
        len(taxonomy.furniture_ids),
        base_channels=8,
    )
    output = model(control)
    assert output["foreground_logits"].shape == (1, 1, 16, 16)
    assert output["class_logits"].shape == (
        1,
        len(taxonomy.furniture_ids),
        16,
        16,
    )
