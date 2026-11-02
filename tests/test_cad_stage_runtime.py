from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.cad_stage_runtime import (
    fuse_junction_evidence,
    fuse_learned_opening_consumer,
    opening_control_maps,
)


def test_opening_control_maps_preserve_foreground_and_family() -> None:
    family_probability = torch.rand(1, 2, 8, 8)
    foreground = torch.rand(1, 8, 8)
    family_logits = torch.randn(1, 2, 8, 8)
    result = opening_control_maps(
        {
            "family_probability": family_probability,
            "foreground": foreground,
            "family_logits": family_logits,
        }
    )
    assert torch.equal(
        result["opening_family_probability"],
        family_probability,
    )
    assert torch.equal(
        result["opening_foreground_probability"],
        foreground,
    )
    assert torch.equal(result["opening_family_logits"], family_logits)


def test_junction_evidence_fuses_geometry_and_structure() -> None:
    geometry = torch.full((1, 8, 8), 0.2)
    structure = torch.full((1, 1, 8, 8), -4.0)
    structure[:, :, :, 4:] = 4.0
    fused = fuse_junction_evidence(geometry, structure)
    assert fused.shape == (1, 8, 8)
    assert fused[:, :, :4].min() >= 0.2
    assert fused[:, :, 4:].min() > 0.9


def test_disabled_opening_consumer_is_bit_exact() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 2}),
            "window": Counter({"Regular Window": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    room_count = len(taxonomy.room_ids)
    channels = room_count + len(taxonomy.opening_ids) + 3
    structure = torch.randn(2, channels, 8, 8)
    opening_logits = torch.randn(2, 1, 8, 8)
    opening_type_logits = torch.randn(2, len(taxonomy.opening_ids), 8, 8)
    fused, family = fuse_learned_opening_consumer(
        structure,
        opening_logits,
        opening_type_logits,
        torch.rand(2, 2, 8, 8),
        taxonomy,
        enabled=False,
    )
    assert torch.equal(fused, structure)
    assert family is None


def test_enabled_opening_consumer_changes_only_opening_channels() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 2}),
            "window": Counter({"Regular Window": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    room_count = len(taxonomy.room_ids)
    channels = room_count + len(taxonomy.opening_ids) + 3
    structure = torch.zeros(1, channels, 8, 8)
    opening_logits = torch.zeros(1, 1, 8, 8)
    opening_type_logits = torch.zeros(
        1,
        len(taxonomy.opening_ids),
        8,
        8,
    )
    fused, family = fuse_learned_opening_consumer(
        structure,
        opening_logits,
        opening_type_logits,
        torch.ones(1, 2, 8, 8),
        taxonomy,
        enabled=True,
    )
    opening_start = room_count + 1
    opening_end = room_count + 2 + len(taxonomy.opening_ids)
    assert torch.equal(fused[:, :opening_start], structure[:, :opening_start])
    assert torch.equal(fused[:, opening_end:], structure[:, opening_end:])
    assert fused[:, room_count + 1].min() > 0.5
    assert family is not None
    assert torch.allclose(family, torch.ones_like(family))
