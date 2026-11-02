from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.masked_control import (
    apply_control_mask,
    control_channel_groups,
    sample_control_mask,
)


def test_masked_control_boundaries_and_groups() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    source = torch.rand(2, taxonomy.num_target_channels, 16, 16)
    footprint = torch.ones(2, 1, 16, 16)
    groups = control_channel_groups(taxonomy)
    assert set(groups) == {"room", "wall", "opening", "furniture"}
    zero = sample_control_mask(
        source,
        footprint,
        taxonomy,
        ratio=0.0,
    )
    assert not zero.value.any()
    full = sample_control_mask(
        source,
        footprint,
        taxonomy,
        ratio=1.0,
    )
    assert full.value.all()


def test_apply_masked_control_uses_missing_token() -> None:
    source = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :2] = True
    missing = torch.tensor([2.0, 3.0, 4.0])
    output = apply_control_mask(source, mask, missing)
    assert torch.all(output[:, :, :2] > 1.5)
    assert torch.all(output[:, :, 2:] == 1.0)
