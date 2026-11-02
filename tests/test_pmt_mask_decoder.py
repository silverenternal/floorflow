from __future__ import annotations

import torch

from floorflow.model.pmt_mask_decoder import PMTMaskDecoderAdapter


def test_pmt_decoder_shapes_probabilities_and_gradients() -> None:
    decoder = PMTMaskDecoderAdapter(
        24,
        7,
        hidden_channels=16,
        heads=4,
        blocks=2,
        grid=8,
        mask_dropout=0.0,
        upscale_steps=1,
        pixel_refiner=True,
    )
    features = torch.randn(2, 24, 16, 16, requires_grad=True)
    output = decoder(features)
    assert output["logits"].shape == (2, 7, 16, 16)
    assert output["room_logits"].shape == (2, 6, 16, 16)
    assert output["mask_logits"].shape == (2, 7, 16, 16)
    assert len(output["aux_mask_logits"]) == 2
    assert output["channel_tokens"].shape == (2, 7, 16)
    assert decoder.pixel_refiner is not None
    assert torch.allclose(
        output["probability"].sum(dim=1),
        torch.ones(2, 16, 16),
        atol=1.0e-5,
    )
    output["room_logits"].mean().backward()
    assert features.grad is not None
    assert decoder.queries.grad is not None


def test_pmt_mask_dropout_keeps_fully_masked_queries_finite() -> None:
    decoder = PMTMaskDecoderAdapter(
        8,
        4,
        hidden_channels=8,
        heads=2,
        blocks=1,
        grid=4,
        mask_dropout=0.75,
        upscale_steps=0,
    )
    decoder.train()
    output = decoder(torch.randn(3, 8, 8, 8))
    assert torch.isfinite(output["logits"]).all()
