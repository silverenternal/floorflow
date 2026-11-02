from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.coarse_support_refiner import (
    CoarseSupportRefiner,
    coarse_support_control,
)


def test_coarse_support_refiner_shapes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.full((1, 32, 32), taxonomy.room_ids[0])
    footprint = torch.ones(1, 32, 32)
    control = coarse_support_control(
        labels,
        footprint,
        taxonomy,
        size=8,
    )
    model = CoarseSupportRefiner(control.shape[1], base_channels=8)
    output = model(control)
    assert output.shape == (1, 1, 16, 16)
