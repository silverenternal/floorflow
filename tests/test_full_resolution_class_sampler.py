import torch

from floorflow.heads.full_resolution_class_sampler import (
    FullResolutionClassSampler,
)


def test_full_resolution_sampler_is_identity_at_initialization() -> None:
    sampler = FullResolutionClassSampler(16, 8, 4)
    query_features = torch.randn(2, 5, 16)
    masks = torch.randn(2, 5, 32, 32)
    class_logits = torch.randn(2, 5, 5)
    features = torch.randn(2, 8, 32, 32)
    output, diagnostics = sampler(
        query_features,
        masks,
        class_logits,
        features,
    )
    assert torch.equal(output, class_logits)
    assert diagnostics["residual_norm"] == 0


def test_full_resolution_sampler_uses_mask_local_features() -> None:
    sampler = FullResolutionClassSampler(16, 8, 4)
    query_features = torch.randn(1, 2, 16)
    masks = torch.full((1, 2, 32, 32), -8.0)
    masks[0, 0, :8, :8] = 8.0
    masks[0, 1, 24:, 24:] = 8.0
    class_logits = torch.randn(1, 2, 5, requires_grad=True)
    features = torch.randn(1, 8, 32, 32, requires_grad=True)
    output, diagnostics = sampler(
        query_features,
        masks,
        class_logits,
        features,
    )
    assert output.shape == class_logits.shape
    assert diagnostics["centroid_x_abs"] > 0.25
    assert diagnostics["centroid_y_abs"] > 0.25
    with torch.no_grad():
        sampler.output.weight.normal_()
    output, diagnostics = sampler(
        query_features,
        masks,
        class_logits,
        features,
    )
    output.square().mean().backward()
    assert features.grad is not None
    assert sampler.feature_projection.weight.grad is not None
    assert sampler.feature_projection.weight.grad.abs().sum() > 0
