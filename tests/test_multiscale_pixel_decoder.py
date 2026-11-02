from __future__ import annotations

import torch

from floorflow.model.multiscale_pixel_decoder import MultiScalePixelDecoder


def test_multiscale_pixel_decoder_builds_three_level_pyramid() -> None:
    decoder = MultiScalePixelDecoder(32, hidden_channels=24, levels=3)
    features = torch.randn(2, 32, 16, 16, requires_grad=True)
    output, pyramid = decoder(features)
    assert output.shape == (2, 24, 16, 16)
    assert len(pyramid) == 3
    assert [level.shape[-2:] for level in pyramid] == [
        (16, 16),
        (8, 8),
        (4, 4),
    ]
    output.mean().backward()
    assert features.grad is not None
