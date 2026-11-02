from __future__ import annotations

import torch

from floorflow.heads.compact_boundary_encoder import CompactBoundaryEncoder


def test_compact_encoder_matches_segmenter_contract_and_backpropagates() -> None:
    encoder = CompactBoundaryEncoder(
        4,
        decoder_channels=12,
        channels=(8, 16),
        blocks=(1, 1),
    )
    footprint = torch.zeros(2, 1, 32, 32)
    footprint[:, :, 4:28, 4:28] = 1.0
    features = encoder.encode_control(footprint)
    logits = encoder(footprint)
    loss = logits.square().mean() + features.square().mean()
    loss.backward()
    assert features.shape == (2, 12, 32, 32)
    assert logits.shape == (2, 4, 32, 32)
    assert encoder.stem[0].weight.grad is not None
    assert torch.isfinite(encoder.stem[0].weight.grad).all()
