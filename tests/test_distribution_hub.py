from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.hub import CoupledDistributionHub


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


def test_distribution_hub_couples_pixel_and_element_outputs() -> None:
    hierarchy = ChannelHierarchy.from_taxonomy(_taxonomy())
    hub = CoupledDistributionHub(
        12,
        hierarchy,
        hidden_channels=16,
        pixel_grid=4,
        element_grid=4,
        queries_per_group={
            "wall": 2,
            "opening": 2,
            "room": 2,
            "furniture": 2,
            "support": 1,
        },
    )
    features = torch.randn(1, 12, 16, 16)
    output, loss, components = hub.loss(
        features,
        footprint=torch.ones(1, 1, 16, 16),
    )
    assert output["pixel"]["probability"].shape[0] == 1
    assert output["elements"]["room"]["mask_logits"].shape[0] == 1
    assert set(output["parent_gates"]) >= {
        "wall",
        "opening",
        "room",
        "room_region",
    }
    assert output["parent_gates"]["room"].shape == (1, 1, 16, 16)
    assert torch.allclose(
        output["parent_gates"]["room_region"],
        1.0 - output["parent_gates"]["wall"],
    )
    assert "distribution_coupling_loss" in components
    loss.backward()
    assert hub.pixel.context_scale.grad is not None
    assert hub.element.group_queries["room"].grad is not None


def test_distribution_hub_honors_element_feedback_initialization() -> None:
    hierarchy = ChannelHierarchy.from_taxonomy(_taxonomy())
    hub = CoupledDistributionHub(
        12,
        hierarchy,
        hidden_channels=16,
        pixel_grid=4,
        element_grid=4,
        element_feedback_init=0.25,
    )
    assert torch.allclose(
        hub.element_feedback_scale,
        torch.full_like(hub.element_feedback_scale, 0.25),
    )
