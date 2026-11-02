import torch

from floorflow.model.layoutbridge_adapters import (
    AnisotropicVelocityRefinement,
    DualStreamStructureInjection,
    laplacian_filter,
    reconstruction_frequency_loss,
)


def test_adapted_blocks_are_identity_at_initialization() -> None:
    features = torch.randn(2, 16, 8, 8)
    guidance = torch.randn(2, 16, 8, 8)
    time_value = torch.rand(2)
    ddi = DualStreamStructureInjection(16, 16)
    mixed, _ = ddi(features, guidance, time_value)
    assert torch.allclose(mixed, features)
    anisotropic = AnisotropicVelocityRefinement(16, 12, 16)
    refined, _ = anisotropic(features, torch.randn(2, 12, 8, 8), time_value)
    assert torch.allclose(refined, features)


def test_frequency_loss_and_laplacian() -> None:
    prediction = torch.randn(2, 4, 16, 16)
    target = torch.randn(2, 4, 16, 16)
    low, high = reconstruction_frequency_loss(prediction, target)
    assert torch.isfinite(low) and torch.isfinite(high)
    assert laplacian_filter(target).shape == target.shape
