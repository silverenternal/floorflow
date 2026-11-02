from __future__ import annotations

import torch

from floorflow.heads.room_mask_decoder import RoomMaskDecoder


def test_full_resolution_prototype_mask_branch_is_live() -> None:
    decoder = RoomMaskDecoder(
        8,
        2,
        hidden_dim=16,
        num_queries=4,
        layers=1,
        heads=4,
        feature_scales=(1,),
        full_resolution_mask_refinement=True,
    )
    with torch.inference_mode():
        highres = torch.randn(1, 2, 32, 32)
    output = decoder(
        torch.randn(1, 8, 8, 8),
        prototype_logits=torch.randn(1, 2, 8, 8),
        full_resolution_prototype_logits=highres,
    )
    assert output["mask_logits"].shape == (1, 4, 32, 32)
    assert output["aux_mask_logits"][-1].shape == (1, 4, 8, 8)
    output["mask_logits"].square().mean().backward()
    assert decoder.full_resolution_mask_scale is not None
    assert decoder.full_resolution_mask_scale.grad is not None
    assert decoder.prototype_projection.weight.grad is not None
