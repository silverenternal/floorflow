from collections import Counter

import numpy as np
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.structure_compositor import (
    StructureReadoutPolicy,
    decode_structure_layers,
    effective_exterior_width,
    owner_cell_plan,
    structure_opening_family_outputs,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Bath": 2}),
            "furniture": Counter(),
            "door": Counter({"Door": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _structure(channels: int):
    return {
        "wall_logits": torch.full((1, 1, 8, 8), -12.0),
        "opening_logits": torch.full((1, 1, 8, 8), -12.0),
        "opening_type_logits": torch.zeros(1, channels, 8, 8),
        "room_boundary_logits": torch.full((1, 1, 8, 8), -12.0),
    }


def test_boundary_gate_removes_ungated_room_seam() -> None:
    taxonomy = _taxonomy()
    room_logits = torch.zeros(1, 2, 8, 8)
    room_logits[:, 0, :, :4] = 8.0
    room_logits[:, 1, :, 4:] = 8.0
    structure = _structure(len(taxonomy.opening_ids))
    footprint = torch.ones(1, 8, 8)
    legacy, _ = decode_structure_layers(
        room_logits,
        structure,
        footprint,
        taxonomy,
        policy=StructureReadoutPolicy(seam_mode="legacy"),
    )
    gated, _ = decode_structure_layers(
        room_logits,
        structure,
        footprint,
        taxonomy,
        policy=StructureReadoutPolicy(seam_mode="boundary_gated"),
    )
    assert legacy.wall.any()
    assert not gated.wall.any()


def test_wall_gate_keeps_supported_room_seam_only() -> None:
    taxonomy = _taxonomy()
    room_logits = torch.zeros(1, 2, 8, 8)
    room_logits[:, 0, :, :4] = 8.0
    room_logits[:, 1, :, 4:] = 8.0
    structure = _structure(len(taxonomy.opening_ids))
    structure["wall_logits"][:, :, :, 4] = -1.0
    none, _ = decode_structure_layers(
        room_logits,
        structure,
        torch.ones(1, 8, 8),
        taxonomy,
        policy=StructureReadoutPolicy(seam_mode="none"),
    )
    gated, _ = decode_structure_layers(
        room_logits,
        structure,
        torch.ones(1, 8, 8),
        taxonomy,
        policy=StructureReadoutPolicy(
            seam_mode="wall_gated",
            room_boundary_threshold=0.2,
        ),
    )
    assert not none.wall.any()
    assert gated.wall.any()


def test_wall_core_gate_trims_low_confidence_extension_tails() -> None:
    taxonomy = _taxonomy()
    room_logits = torch.zeros(1, 2, 8, 8)
    room_logits[:, 0, :, :4] = 8.0
    room_logits[:, 1, :, 4:] = 8.0
    structure = _structure(len(taxonomy.opening_ids))
    structure["wall_logits"][:] = -5.0
    structure["wall_logits"][:, :, 3:5, 1:7] = 3.0
    structure["wall_logits"][:, :, 3:5, 1:3] = 0.0
    policy = StructureReadoutPolicy(
        seam_mode="none",
        wall_core_threshold=0.75,
        wall_tail_radius=0,
    )
    layers, _ = decode_structure_layers(
        room_logits,
        structure,
        torch.ones(1, 8, 8),
        taxonomy,
        policy=policy,
    )
    assert layers.wall[:, 3:5, 3:7].all()
    assert not layers.wall[:, :, 1:3].any()


def test_junction_degree_and_opening_host_filter_are_consumed() -> None:
    taxonomy = _taxonomy()
    room_logits = torch.zeros(1, 2, 8, 8)
    room_logits[:, 0] = 8.0
    structure = _structure(len(taxonomy.opening_ids))
    structure["junction_logits"] = torch.zeros(1, 1, 8, 8)
    degree = torch.full((1, 5, 8, 8), -8.0)
    degree[:, 1] = 8.0
    structure["junction_degree_logits"] = degree
    structure["opening_logits"] = torch.zeros(1, 1, 8, 8)
    layers, _ = decode_structure_layers(
        room_logits,
        structure,
        torch.ones(1, 8, 8),
        taxonomy,
        policy=StructureReadoutPolicy(
            seam_mode="none",
            junction_degree_min=2,
        ),
    )
    assert not layers.wall.any()
    assert not layers.opening.any()


def test_exterior_width_scales_with_resolution() -> None:
    policy = StructureReadoutPolicy(
        exterior_mode="resolution_scaled",
        exterior_width=8,
        reference_resolution=384,
    )
    assert effective_exterior_width((384, 384), policy) == 8
    assert effective_exterior_width((192, 192), policy) == 4


def test_owner_cell_plan_uses_shared_cell_source() -> None:
    taxonomy = _taxonomy()
    labels = np.zeros((16, 16), dtype=np.int64)
    labels[:, :8] = taxonomy.room_ids[0]
    labels[:, 8:] = taxonomy.room_ids[1]
    labels[:, 7:9] = taxonomy.class_id("wall")
    labels[6:10, 7:9] = taxonomy.opening_ids[0]
    plan = owner_cell_plan(
        labels,
        np.ones((16, 16), dtype=bool),
        taxonomy,
        minimum_area=4,
    )
    assert plan.room_source == "owner_cell_complex"
    assert plan.rooms
    assert plan.walls
    assert {wall.wall_type for wall in plan.walls} <= {
        "interior",
        "exterior",
    }
    assert all(
        0 <= opening.wall_index < len(plan.walls)
        for opening in plan.openings
    )


def test_structure_openings_are_exposed_to_interval_reader() -> None:
    structure = _structure(2)
    probability, logits = structure_opening_family_outputs(
        structure,
        (16, 16),
    )
    assert probability.shape == (1, 16, 16)
    assert logits.shape == (1, 2, 16, 16)
