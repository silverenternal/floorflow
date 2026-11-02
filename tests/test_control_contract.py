from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.control_cache import write_control_cache
from floorflow.data.control_contract import (
    control_provenance,
    expected_control_shapes,
    oracle_control_payload,
    summarize_control_file,
)
from floorflow.data.taxonomy import FloorplanTaxonomy

TAXONOMY = Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json")


def test_oracle_control_contract_shapes_and_provenance() -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    room = taxonomy.room_ids[0]
    labels = torch.full((1, 32, 32), room, dtype=torch.long)
    footprint = torch.ones(1, 32, 32)

    controls = oracle_control_payload(
        labels,
        footprint,
        taxonomy,
        room_size=16,
        structure_size=8,
        furniture_size=12,
        support_size=12,
    )
    expected = expected_control_shapes(
        taxonomy,
        room_size=16,
        structure_size=8,
        furniture_size=12,
        support_size=12,
    )
    assert {
        key: tuple(value.shape[1:]) for key, value in controls.items()
    } == expected
    assert controls["program"].dtype == torch.bool
    assert int(controls["program"].sum()) == 1
    assert control_provenance("oracle") == {
        "target_used_for_control": True,
        "deployable": False,
    }


def test_control_contract_round_trip_and_summary(tmp_path: Path) -> None:
    taxonomy = FloorplanTaxonomy.read(TAXONOMY)
    room = taxonomy.room_ids[0]
    labels = torch.full((1, 32, 32), room, dtype=torch.long)
    footprint = torch.ones(1, 32, 32)
    controls = oracle_control_payload(
        labels,
        footprint,
        taxonomy,
        room_size=16,
        structure_size=8,
        furniture_size=12,
        support_size=12,
    )
    path = tmp_path / "sample.npz"
    write_control_cache(
        path,
        room_logits=controls["room_logits"],
        program=controls["program"],
        structure=controls["structure"],
        furniture=controls["furniture"],
        support=controls["support"],
        room_size=16,
    )

    summary = summarize_control_file(
        path,
        taxonomy,
        room_size=16,
        structure_size=8,
        furniture_size=12,
        support_size=12,
    )
    assert summary["sha256"]
    assert summary["arrays"]["program"]["count"] == 1
    assert summary["arrays"]["room_logits"]["shape"] == [
        len(taxonomy.room_ids),
        16,
        16,
    ]
