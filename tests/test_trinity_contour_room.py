from __future__ import annotations

import torch

from floorflow.model.trinity_contour_room import (
    ContourTrinityRoomModel,
    FeatureTrinityRoomModel,
)


def test_contour_trinity_room_model_shapes_and_gradients() -> None:
    model = ContourTrinityRoomModel(
        hidden=32,
        depth=2,
        heads=4,
        room_tokens=8,
        grid=4,
    )
    footprint = torch.ones(2, 1, 32, 32)
    z_t = torch.randn(2, 8, 3)
    time = torch.tensor([0.25, 0.75])
    prediction, diagnostics = model.forward_with_aux(z_t, time, footprint)
    assert prediction.shape == z_t.shape
    assert diagnostics["token_logits"].shape == (2, 8)
    prediction.mean().backward()
    assert model.denoiser.final.linear.weight.grad is not None


def test_feature_conditioned_room_model_accepts_backbone_features() -> None:
    model = FeatureTrinityRoomModel(
        48,
        hidden=32,
        depth=2,
        heads=4,
        room_tokens=8,
        grid=4,
        num_room_classes=5,
    )
    features = torch.randn(2, 48, 16, 16)
    z_t = torch.randn(2, 8, 3)
    time = torch.tensor([0.2, 0.8])
    prediction, diagnostics = model.forward_with_aux(z_t, time, features)
    assert prediction.shape == z_t.shape
    assert diagnostics["class_logits"].shape == (2, 8, 5)
    prediction.mean().backward()
    assert model.encoder.project[0].weight.grad is not None
