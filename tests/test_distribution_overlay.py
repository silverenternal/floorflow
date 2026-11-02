from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.inference.distribution_overlay import (
    DistributionOverlaySettings,
    compose_distribution_overlay,
    element_family_scores,
    overlay_room_labels,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_overlay_is_deterministic_and_keeps_layers_separate() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    footprint = torch.ones(1, 8, 8, dtype=torch.long)
    elements = {}
    for name, count in (
        ("wall", 1),
        ("opening", 1),
        ("room", 1),
        ("furniture", 1),
        ("support", 1),
    ):
        class_count = len(hierarchy.groups[name])
        mask = torch.full((1, 1, 8, 8), -10.0)
        mask[:, :, 2:6, 2:6] = 10.0
        types = torch.full((1, 1, class_count + 1), -10.0)
        types[..., 0] = 10.0
        elements[name] = {"mask_logits": mask, "type_logits": types}
    scores = element_family_scores(elements, hierarchy, (8, 8))
    pixel_logits = torch.zeros(1, taxonomy.num_classes, 8, 8)
    rooms = overlay_room_labels(
        pixel_logits,
        scores["room"],
        hierarchy.groups["room"],
        footprint,
        DistributionOverlaySettings(),
    )
    labels, layers = compose_distribution_overlay(
        rooms,
        scores,
        footprint,
        taxonomy,
        DistributionOverlaySettings(),
        pixel_labels=torch.zeros_like(rooms),
    )
    assert labels.shape == footprint.shape
    assert layers.room_labels.shape == labels.shape
    assert torch.equal(labels, layers.visible_labels())
    assert layers.wall.shape == labels.shape
