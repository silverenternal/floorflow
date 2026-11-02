from __future__ import annotations

import torch

from floorflow.heads.attention_masks import soft_owner_attention_bias
from floorflow.heads.room_mask_decoder import RoomMaskDecoder


def test_room_mask_decoder_uses_recoverable_float_attention(
    monkeypatch,
) -> None:
    decoder = RoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=4,
        layers=2,
        heads=4,
        feature_scales=(1,),
    )
    captured = []
    layer = decoder.layers[1]
    original = layer.forward

    def capture(queries, memory, attention_mask):
        captured.append(attention_mask.detach().clone())
        return original(queries, memory, attention_mask)

    monkeypatch.setattr(layer, "forward", capture)
    output = decoder(
        torch.randn(1, 8, 8, 8),
        prototype_logits=torch.randn(1, 3, 8, 8),
    )
    assert captured
    assert captured[0].is_floating_point()
    assert torch.isfinite(captured[0]).all()
    output["mask_logits"].sum().backward()
    assert decoder.mask_head[-1].weight.grad is not None


def test_soft_owner_bias_accepts_token_and_spatial_layouts() -> None:
    token_logits = torch.randn(1, 4, 16)
    spatial_logits = token_logits.reshape(1, 4, 4, 4)
    token_bias = soft_owner_attention_bias(token_logits)
    spatial_bias = soft_owner_attention_bias(spatial_logits)
    assert token_bias.shape == token_logits.shape
    assert spatial_bias.shape == (1, 4, 16)
    assert torch.equal(token_bias, spatial_bias)
