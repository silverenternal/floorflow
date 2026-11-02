import torch

from floorflow.heads.adaptive_query_reference import (
    AdaptiveQueryReferenceRefiner,
    prototype_reference_points,
)


def test_adaptive_reference_sampling_is_image_conditioned() -> None:
    refiner = AdaptiveQueryReferenceRefiner(
        query_dim=16,
        feature_channels=8,
        hidden_dim=16,
    )
    queries = torch.randn(2, 8, 16)
    features = torch.randn(2, 8, 12, 12)
    update, diagnostics = refiner(queries, features)
    assert update.shape == queries.shape
    assert diagnostics["reference_points"].shape == (2, 8, 2)
    assert diagnostics["reference_samples"].shape == (2, 8, 4, 2)
    assert torch.all(diagnostics["reference_points"] >= 0.0)
    assert torch.all(diagnostics["reference_points"] <= 1.0)
    (update.square().mean()).backward()
    assert refiner.feature_projection.weight.grad is not None
    assert refiner.reference_head[-1].weight.grad is not None


def test_adaptive_reference_is_near_identity_then_image_dependent() -> None:
    refiner = AdaptiveQueryReferenceRefiner(
        query_dim=16,
        feature_channels=8,
        hidden_dim=16,
    )
    queries = torch.randn(2, 8, 16)
    first_features = torch.zeros(2, 8, 12, 12)
    second_features = torch.ones(2, 8, 12, 12)
    initial, initial_diagnostics = refiner(queries, first_features)
    assert torch.allclose(initial, torch.zeros_like(initial), atol=1.0e-2)
    _, initial_second = refiner(queries, second_features)
    assert torch.allclose(
        initial_diagnostics["reference_points"],
        initial_second["reference_points"],
        atol=1.0e-2,
    )
    with torch.no_grad():
        refiner.reference_head[-1].weight.normal_(0.0, 0.1)
    _, first = refiner(queries, first_features)
    _, second = refiner(queries, second_features)
    assert not torch.equal(
        first["reference_points"],
        second["reference_points"],
    )


def test_live_reference_residual_opens_first_step_gradient() -> None:
    refiner = AdaptiveQueryReferenceRefiner(
        query_dim=16,
        feature_channels=8,
        hidden_dim=16,
        live_reference_residual_scale=0.05,
    )
    queries = torch.randn(2, 8, 16)
    features = torch.randn(2, 8, 12, 12)
    update, _diagnostics = refiner(queries, features)
    update.square().mean().backward()
    assert refiner.reference_head[-1].weight.grad is not None
    assert refiner.reference_head[-1].weight.grad.abs().sum() > 0.0


def test_prototype_saliency_moves_cell_anchors_toward_rooms() -> None:
    logits = torch.full((1, 3, 8, 8), -8.0)
    logits[:, 0, 6:, 6:] = 8.0
    anchors = prototype_reference_points(logits, 4)
    assert anchors.shape == (1, 4, 2)
    assert anchors[0, 3, 0] > 0.8
    assert anchors[0, 3, 1] > 0.8
    refiner = AdaptiveQueryReferenceRefiner(16, 8)
    _update, diagnostics = refiner(
        torch.randn(1, 4, 16),
        torch.randn(1, 8, 12, 12),
        reference_logits=logits,
    )
    assert diagnostics["reference_prior_shift"] > 0.0
