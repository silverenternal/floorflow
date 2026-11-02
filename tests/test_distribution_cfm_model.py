from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.hierarchy import ChannelHierarchy


class _Output:
    in_channels = 8


class _Base(nn.Module):
    target_channels = 7

    def __init__(self) -> None:
        super().__init__()
        self.output = _Output()
        self.proj = nn.Conv2d(7, 8, 1)

    def _encode_features(self, value, time, footprint, floor_ids=None):
        del time, floor_ids
        return self.proj(value) + footprint.float()

    def _velocity_from_features(self, features, time=None):
        del time
        return features[:, :7]

    def forward_with_aux(self, value, time, footprint, floor_ids=None):
        del time, floor_ids
        velocity = self.proj(value)[:, :7]
        return velocity, {"room_partition_logits": features_room(value)}


def features_room(value):
    return value[:, :1].expand(-1, 2, -1, -1)


def _taxonomy():
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    return taxonomy


def test_distribution_cfm_model_shares_backbone_outputs() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    model = DistributionCFMModel(
        _Base(),
        hierarchy,
        hidden_channels=8,
        pixel_grid=4,
        element_grid=4,
        queries_per_group={
            "wall": 1,
            "opening": 1,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
    )
    x_t = torch.randn(1, 7, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    velocity, auxiliary = model.forward_with_aux(
        x_t,
        torch.zeros(1),
        footprint,
    )
    assert velocity.shape == x_t.shape
    assert auxiliary == {}
    contour = model.contour_initial_state(footprint)
    assert contour["pixel"]["logits"].shape[1] == taxonomy.num_classes
    assert contour["elements"]["room"]["mask_logits"].shape[1] == 2
    assert contour["cfm"]["state"].shape == x_t.shape


def test_distribution_cfm_model_propagates_element_coupling_initialization() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    model = DistributionCFMModel(
        _Base(),
        hierarchy,
        hidden_channels=8,
        pixel_grid=4,
        element_grid=4,
        element_blend_init=0.5,
        element_feedback_init=-0.2,
    )
    assert model.element_blend_init == 0.5
    assert model.element_feedback_init == -0.2
    assert torch.allclose(model.cfm_bridge.element_blend, torch.full((7,), 0.5))
    assert torch.allclose(
        model.hub.element_feedback_scale,
        torch.full((7,), -0.2),
    )


def test_style_residuals_reach_pixel_element_and_velocity_paths() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    model = DistributionCFMModel(
        _Base(),
        hierarchy,
        hidden_channels=8,
        pixel_grid=4,
        element_grid=4,
        queries_per_group={
            "wall": 1,
            "opening": 1,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
        num_styles=2,
        style_rank=2,
    )
    footprint = torch.ones(2, 1, 16, 16)
    style_ids = torch.tensor([0, 1])
    initial = model.contour_initial_state(
        footprint,
        style_ids=style_ids,
    )
    with torch.no_grad():
        model.hub.pixel.style_pixel_residual.class_bias.weight.normal_(std=0.1)
        model.hub.element.style_element_residual.presence_bias.weight.normal_(
            std=0.1
        )
    styled = model.contour_initial_state(
        footprint,
        style_ids=style_ids,
    )
    loss = (
        styled["pixel"]["logits"][:, 1:].square().sum()
        + styled["elements"]["room"]["presence_logits"].square().sum()
        + model(
            torch.zeros(2, 7, 16, 16),
            torch.zeros(2),
            footprint,
            style_ids=style_ids,
        ).square().sum()
    )
    loss.backward()
    assert initial["pixel"]["logits"].shape == styled["pixel"]["logits"].shape
    assert (
        model.hub.pixel.style_pixel_residual.class_bias.weight.grad is not None
    )
    assert (
        model.hub.element.style_element_residual.presence_bias.weight.grad
        is not None
    )
    assert model.velocity_style_adapter.embedding.weight.grad is not None


def test_source_aware_velocity_refiner_starts_as_identity_and_receives_grad() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    model = DistributionCFMModel(
        _Base(),
        hierarchy,
        hidden_channels=8,
        pixel_grid=4,
        element_grid=4,
        queries_per_group={
            "wall": 1,
            "opening": 1,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
        velocity_refiner_hidden=8,
    )
    x_t = torch.randn(2, 7, 16, 16)
    source = torch.randn_like(x_t)
    footprint = torch.ones(2, 1, 16, 16)
    time = torch.full((2,), 0.25)
    base = model(x_t, time, footprint)
    refined = model(x_t, time, footprint, source=source)
    assert torch.equal(base, refined)
    with torch.no_grad():
        model.velocity_refiner.output.weight.normal_(std=0.01)
    corrected = model(x_t, time, footprint, source=source)
    assert not torch.equal(base, corrected)
    corrected.square().mean().backward()
    assert model.velocity_refiner.output.weight.grad is not None
