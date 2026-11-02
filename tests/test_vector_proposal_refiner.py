"""Tests for vector proposal residual refinement."""

from __future__ import annotations

import torch

from floorflow.heads.vector_proposal_refiner import VectorProposalRefiner


def test_vector_proposal_refiner_preserves_shapes_and_masks() -> None:
    refiner = VectorProposalRefiner(in_channels=12, hidden_dim=32)
    features = torch.randn(2, 12, 16, 16, requires_grad=True)
    rooms = torch.rand(2, 5, 8, 2)
    room_valid = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    walls = torch.rand(2, 7, 2, 2)
    wall_valid = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    openings = torch.rand(2, 6, 2)
    opening_valid = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )

    output = refiner(
        features,
        rooms,
        room_valid,
        walls,
        wall_valid,
        openings,
        opening_valid,
        torch.randn(2, 32),
    )
    loss = sum(
        output[key].float().square().mean()
        for key in (
            "room_polygons",
            "wall_endpoints",
            "opening_centers",
        )
    )
    loss.backward()

    assert output["room_polygons"].shape == rooms.shape
    assert output["wall_endpoints"].shape == walls.shape
    assert output["opening_centers"].shape == openings.shape
    assert torch.count_nonzero(output["room_residual"][0, 3:]) == 0
    assert torch.count_nonzero(output["wall_residual"][0, 3:]) == 0
    assert torch.count_nonzero(output["opening_residual"][0, 2:]) == 0
    assert features.grad is not None
