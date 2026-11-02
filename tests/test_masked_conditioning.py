from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.training.masked_conditioning import prepare_masked_condition


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )


def test_zero_hints_never_exposes_target_control() -> None:
    taxonomy = _taxonomy()
    source = torch.ones(1, taxonomy.num_target_channels, 8, 8)
    footprint = torch.ones(1, 1, 8, 8)
    missing = torch.zeros(taxonomy.num_target_channels)
    prepared = prepare_masked_condition(
        source,
        footprint,
        taxonomy,
        missing,
        missing_ratio=1.0,
        floor_ids=torch.zeros(1, dtype=torch.long),
    )
    assert prepared.view.is_target_control_free
    assert not prepared.control.any()
    assert prepared.missing_mask.all()
    assert torch.all(prepared.coverage == 1.0)


def test_full_control_preserves_layout_and_is_marked_as_such() -> None:
    taxonomy = _taxonomy()
    source = torch.ones(1, taxonomy.num_target_channels, 8, 8)
    footprint = torch.ones(1, 1, 8, 8)
    missing = torch.zeros(taxonomy.num_target_channels)
    prepared = prepare_masked_condition(
        source,
        footprint,
        taxonomy,
        missing,
        missing_ratio=0.0,
    )
    assert prepared.view.mode == "layout_conditioned"
    assert torch.equal(prepared.control, source)
    assert not prepared.missing_mask.any()


def test_sparse_target_mutation_does_not_change_hidden_input() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones(1, 1, 8, 8)
    missing = torch.zeros(taxonomy.num_target_channels)
    first = torch.ones(1, taxonomy.num_target_channels, 8, 8)
    kwargs = {
        "missing_ratio": 0.25,
        "floor_ids": torch.zeros(1, dtype=torch.long),
    }
    torch.manual_seed(11)
    first_prepared = prepare_masked_condition(
        first,
        footprint,
        taxonomy,
        missing,
        **kwargs,
    )
    hidden = first_prepared.missing_mask
    second = first.clone()
    second[hidden] = 17.0
    torch.manual_seed(11)
    second_prepared = prepare_masked_condition(
        second,
        footprint,
        taxonomy,
        missing,
        **kwargs,
    )
    assert torch.equal(
        first_prepared.control[~hidden],
        second_prepared.control[~hidden],
    )
    assert not first_prepared.control[hidden].any()
