"""Tests for projected-control opening query tokens."""

import torch

from floorflow.heads.opening_query_encoder import OpeningQueryEncoder


def test_opening_query_encoder_shapes_and_gradients() -> None:
    encoder = OpeningQueryEncoder(12, hidden_dim=16)
    features = torch.randn(2, 12, 16, 16, requires_grad=True)
    centers = torch.rand(2, 4, 2)
    widths = torch.rand(2, 4)
    family = torch.randn(2, 4, 2, requires_grad=True)
    valid = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]],
        dtype=torch.bool,
    )
    tokens = encoder(features, centers, widths, family, valid)
    tokens.square().mean().backward()
    assert tokens.shape == (2, 4, 16)
    assert torch.equal(tokens[0, 3], torch.zeros(16))
    assert features.grad is not None
    assert family.grad is not None


def test_opening_query_encoder_handles_no_openings() -> None:
    tokens = OpeningQueryEncoder(8, hidden_dim=16)(
        torch.randn(1, 8, 16, 16),
        torch.zeros(1, 0, 2),
        torch.zeros(1, 0),
        torch.zeros(1, 0, 2),
        torch.zeros(1, 0, dtype=torch.bool),
    )
    assert tokens.shape == (1, 0, 16)
