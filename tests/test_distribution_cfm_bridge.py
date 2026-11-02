from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.cfm_bridge import DistributionCFMBridge
from floorflow.distribution.hierarchy import ChannelHierarchy


def test_distribution_cfm_bridge_starts_from_pixel_state() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    bridge = DistributionCFMBridge(hierarchy, element_blend_init=0.0)
    probability = torch.rand(1, taxonomy.num_classes, 8, 8)
    probability = probability / probability.sum(dim=1, keepdim=True)
    pixel = {"probability": probability}
    elements = {
        name: {
            "mask_logits": torch.randn(1, 2, 8, 8),
            "type_logits": torch.randn(1, 2, len(indices) + 1),
        }
        for name, indices in hierarchy.groups.items()
        if indices
    }
    footprint = torch.ones(1, 1, 8, 8)
    output = bridge(pixel, elements, footprint)
    assert torch.allclose(output["state"], output["pixel_state"], atol=1.0e-5)
    output["state"].mean().backward()
    assert bridge.element_blend.grad is not None
