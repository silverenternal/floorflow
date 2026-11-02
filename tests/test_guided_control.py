from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.guided_control import (
    guided_source_from_labels,
    target_channels_from_coarse_controls,
)


def test_guided_source_shapes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    labels = torch.full((1, 32, 32), taxonomy.room_ids[0])
    footprint = torch.ones(1, 32, 32)
    source = guided_source_from_labels(
        labels,
        footprint,
        taxonomy,
        room_size=16,
        structure_size=16,
        furniture_size=16,
    )
    assert source.shape == (1, taxonomy.num_target_channels, 32, 32)
    assert float(source[:, taxonomy.room_ids[0] - 1].mean()) > 0.0


def test_coarse_control_channel_mapping() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    rooms = torch.randn(1, len(taxonomy.room_ids), 16, 16)
    structure = torch.rand(
        1,
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        16,
        16,
    )
    furniture = torch.rand(
        1,
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        16,
        16,
    )
    source = target_channels_from_coarse_controls(
        rooms,
        structure,
        furniture,
        torch.ones(1, 16, 16),
        taxonomy,
    )
    assert source.shape[1] == taxonomy.num_target_channels
    assert torch.isfinite(source).all()
