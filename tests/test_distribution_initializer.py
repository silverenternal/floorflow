from __future__ import annotations

import torch
import torch.nn as nn

from floorflow.model.distribution_initializer import (
    blend_distribution_state,
    distribution_initial_state,
)


class _Base(nn.Module):
    target_channels = 3

    def _encode_features(self, value, time, footprint, floor_ids=None):
        del time, floor_ids
        return value + footprint.float()


class _Head(nn.Module):
    def forward(self, features):
        background = features[:, :1]
        foreground = features[:, :3]
        probability = torch.softmax(
            torch.cat([background, foreground], dim=1),
            dim=1,
        )
        return {
            "probability": probability,
            "reliability": probability.amax(dim=1, keepdim=True),
        }


def test_distribution_initial_state_is_contour_masked() -> None:
    footprint = torch.zeros(1, 1, 8, 8)
    footprint[:, :, 2:6, 2:6] = 1
    output = distribution_initial_state(
        _Base(),
        _Head(),
        footprint,
    )
    assert output["state"].shape == (1, 3, 8, 8)
    assert output["state"][footprint.expand_as(output["state"]) == 0].abs().max() == 0


def test_blend_distribution_state_supports_baseline_endpoints() -> None:
    control = torch.ones(1, 3, 4, 4)
    distribution = torch.zeros(1, 3, 4, 4)
    footprint = torch.ones(1, 1, 4, 4)
    assert torch.equal(
        blend_distribution_state(
            control,
            distribution,
            footprint,
            blend=0.0,
        ),
        control,
    )
    assert torch.equal(
        blend_distribution_state(
            control,
            distribution,
            footprint,
            blend=1.0,
        ),
        distribution,
    )
