from __future__ import annotations

import torch
import torch.nn as nn

from floorflow.model.distribution_controller import PixelDistributionController
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
)


class _DummyOutput:
    in_channels = 24


class _DummyBase(nn.Module):
    target_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.output = _DummyOutput()
        self.proj = nn.Conv2d(4, 24, 3, padding=1)

    def _encode_features(self, x_t, time, footprint, floor_ids=None):
        del time, floor_ids
        return self.proj(x_t) * footprint.float()

    def _velocity_from_features(self, features, time):
        del time
        return features[:, :4]

    def forward_with_aux(self, x_t, time, footprint, floor_ids=None):
        velocity = self.proj(x_t)[:, :4] * footprint.float()
        return velocity, {
            "room_partition_logits": torch.zeros(1, 3, 16, 16),
            "structured_objects": {"wall_logits": torch.zeros(1, 1, 16, 16)},
        }


def test_controller_is_zero_initialized_and_preserves_base_outputs() -> None:
    base = _DummyBase()
    controller = PixelDistributionController(
        base,
        num_classes=5,
        room_ids=(2, 3, 4),
        hidden_channels=12,
    )
    x_t = torch.randn(1, 4, 16, 16)
    footprint = torch.ones(1, 1, 16, 16)
    velocity, auxiliary = controller.forward_with_aux(
        x_t,
        torch.zeros(1),
        footprint,
    )
    base_velocity, base_aux = base.forward_with_aux(
        x_t,
        torch.zeros(1),
        footprint,
    )
    assert torch.allclose(velocity, base_velocity)
    assert torch.allclose(
        auxiliary["room_partition_logits"],
        base_aux["room_partition_logits"],
    )
    assert "pixel_distribution" in auxiliary


def test_controller_accepts_masked_channel_distribution_head() -> None:
    base = _DummyBase()
    head = MaskedChannelDistributionHead(
        24,
        5,
        hidden_channels=16,
        heads=4,
        layers=1,
        grid=8,
    )
    controller = PixelDistributionController(
        base,
        num_classes=5,
        room_ids=(2, 3, 4),
        distribution_head=head,
    )
    velocity, auxiliary = controller.forward_with_aux(
        torch.randn(1, 4, 16, 16),
        torch.zeros(1),
        torch.ones(1, 1, 16, 16),
    )
    assert velocity.shape == (1, 4, 16, 16)
    assert auxiliary["pixel_distribution"]["logits"].shape == (1, 5, 16, 16)


def test_learned_velocity_gate_is_zero_initialized_and_trainable() -> None:
    base = _DummyBase()
    controller = PixelDistributionController(
        base,
        num_classes=5,
        room_ids=(2, 3, 4),
        hidden_channels=12,
        velocity_residual=True,
        gate_hidden=16,
    )
    x_t = torch.randn(1, 4, 16, 16)
    time = torch.zeros(1)
    footprint = torch.ones(1, 1, 16, 16)
    gated_velocity, auxiliary = controller.forward_with_aux(
        x_t,
        time,
        footprint,
    )
    base_velocity, _ = base.forward_with_aux(x_t, time, footprint)
    assert torch.allclose(gated_velocity, base_velocity)
    assert auxiliary["distribution_velocity_gate"].shape == x_t.shape
    auxiliary["distribution_velocity_gate"].mean().backward()
    assert controller.velocity_gate[-1].weight.grad is not None
