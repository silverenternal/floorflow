from __future__ import annotations

import torch

from floorflow.heads.room_mask_decoder import RoomMaskDecoder


def test_mask_region_evidence_reaches_room_class_logits() -> None:
    decoder = RoomMaskDecoder(
        8,
        3,
        hidden_dim=16,
        num_queries=2,
        layers=1,
        heads=4,
        feature_scales=(1,),
    )
    prototype_logits = torch.full((1, 3, 8, 8), -4.0)
    prototype_logits[:, 0, :, :4] = 4.0
    prototype_logits[:, 1, :, 4:] = 4.0
    prototype_logits.requires_grad_()
    with torch.no_grad():
        decoder.class_head.weight.zero_()
        decoder.class_head.bias.zero_()
        decoder.mask_head[-1].weight.normal_(std=0.1)
        decoder.fusion_scale.fill_(0.5)
    output = decoder(
        torch.randn(1, 8, 8, 8),
        prototype_logits=prototype_logits,
    )
    evidence = output["mask_class_evidence"]
    assert evidence is not None
    assert evidence.shape == (1, 2, 3)
    assert output["class_logits"][:, :, :3].abs().sum() > 0
    output["class_logits"][:, :, 0].sum().backward()
    assert decoder.fusion_scale.grad is not None
    assert decoder.fusion_scale.grad.abs() > 0
    assert prototype_logits.grad is not None
