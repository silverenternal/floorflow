from __future__ import annotations

import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.representation.layered_floorplan import LayeredFloorplan


def test_layered_output_preserves_room_branch() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    room_id = taxonomy.room_ids[0]
    wall_id = taxonomy.class_id("wall")
    footprint = torch.ones(1, 4, 4, dtype=torch.long)
    rooms = torch.full_like(footprint, room_id)
    wall = torch.zeros_like(footprint)
    wall[:, :, 2] = 1
    layers = LayeredFloorplan(
        room_labels=rooms,
        wall=wall,
        opening=torch.zeros_like(footprint),
        furniture=torch.zeros_like(footprint),
        footprint=footprint,
        taxonomy=taxonomy,
    )
    assert torch.equal(layers.room_labels, rooms)
    assert layers.room_support().all()
    assert (layers.visible_labels()[0, :, 2] == wall_id).all()


def test_program_gate_removes_absent_room_classes() -> None:
    taxonomy = FloorplanTaxonomy.read(
        "outputs/cache/cubicasa5k_full_v2/taxonomy.json"
    )
    first, second = taxonomy.room_ids[:2]
    labels = torch.tensor([[[first, second, first]]])
    program = torch.tensor([[1.0] + [0.0] * (len(taxonomy.room_ids) - 1)])
    gated = gate_room_labels_to_program(labels, program, taxonomy.room_ids)
    assert gated.tolist() == [[[first, 0, first]]]
