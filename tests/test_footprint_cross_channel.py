from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.cfm_model import DistributionCFMModel
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.model.footprint_cross_channel import FootprintCrossChannelPrior


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


def test_cross_channel_prior_is_zero_initialized_and_trainable() -> None:
    prior = FootprintCrossChannelPrior(
        8,
        8,
        {
            "wall": 2,
            "opening": 2,
            "room": 2,
            "furniture": 2,
            "support": 1,
        },
        token_grid=4,
        heads=2,
    )
    features = torch.randn(1, 8, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    output = prior(features, footprint)
    assert all(token.abs().sum() == 0 for token in output["tokens"].values())
    for name in prior.GROUP_ORDER:
        prior.residual_scales[name].data.fill_(1.0)
    changed = prior(features, footprint)["tokens"]
    loss = sum(token.square().sum() for token in changed.values())
    loss.backward()
    assert prior.queries["room"].grad is not None


def test_boundary_cross_attention_is_zero_initialized() -> None:
    prior = FootprintCrossChannelPrior(
        8,
        8,
        {"wall": 2, "opening": 2, "room": 2, "furniture": 2, "support": 1},
        token_grid=4,
        heads=2,
    )
    assert torch.equal(prior.boundary_scale, torch.zeros(1))
    features = torch.randn(1, 8, 16, 16)
    footprint = torch.zeros(1, 1, 16, 16)
    footprint[:, :, 4:12, 4:12] = 1.0
    base = prior(features, footprint)
    for name in prior.GROUP_ORDER:
        prior.residual_scales[name].data.fill_(1.0)
    identity = prior(features, footprint)["tokens"]
    prior.boundary_scale.data.fill_(1.0)
    activated = prior(features, footprint)["tokens"]
    assert any(
        not torch.allclose(identity[name], activated[name])
        for name in prior.GROUP_ORDER
    )
    assert float(base["diagnostics"]["wall"]["boundary_residual_scale"]) == 0.0


def test_overlay_decoder_runs_without_retrieval_memory() -> None:
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
    footprint = torch.ones(1, 1, 16, 16)
    decoded = decode_overlay(
        model,
        None,
        None,
        footprint,
        taxonomy,
        retrieval_prior_weight=0.0,
    )
    assert decoded["retrieval_used"] is False
    assert decoded["labels"].shape == (1, 16, 16)
    assert torch.equal(
        decoded["labels"] * footprint[:, 0].long(),
        decoded["labels"],
    )
