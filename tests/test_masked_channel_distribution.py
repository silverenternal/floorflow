from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.heads.attention_masks import soft_pixel_attention_bias
from floorflow.heads.masked_channel_distribution import (
    MaskedChannelDistributionHead,
)


def test_masked_channel_distribution_shapes_and_gradients() -> None:
    head = MaskedChannelDistributionHead(
        32,
        9,
        hidden_channels=16,
        heads=4,
        layers=2,
        grid=8,
    )
    features = torch.randn(2, 32, 16, 16, requires_grad=True)
    output = head(features)
    assert output["logits"].shape == (2, 9, 16, 16)
    assert torch.allclose(
        output["probability"].sum(dim=1),
        torch.ones(2, 16, 16),
        atol=1.0e-5,
    )
    assert output["channel_tokens"].shape == (2, 9, 16)
    output["room_logits"].mean().backward()
    assert features.grad is not None
    assert head.channel_queries.grad is not None


def test_soft_pixel_attention_bias_allows_recovery() -> None:
    logits = torch.full((1, 2, 8, 8), -8.0)
    logits[:, :, 2:6, 2:6] = 8.0

    bias = soft_pixel_attention_bias(logits, (8, 8))

    assert bias is not None
    assert bias.shape == (1, 2, 64)
    assert torch.isfinite(bias).all()
    assert not bias.eq(float("-inf")).any()
    probability = F.adaptive_avg_pool2d(logits.sigmoid(), (8, 8)).flatten(2)
    assert bias.max() > 0.0
    assert bias[probability <= 0.05].abs().max() < 1.0e-6
