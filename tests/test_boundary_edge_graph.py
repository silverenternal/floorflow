from __future__ import annotations

import torch

from floorflow.geometry.boundary_edge_graph import (
    BoundaryEdgeGraphEncoder,
    boundary_edge_features,
)


def test_boundary_edges_encode_segments_and_preserve_validity() -> None:
    points = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ]
    )
    valid = torch.tensor([[True, True, True, True, False]])
    features, edge_valid = boundary_edge_features(points, valid)
    assert features.shape == (1, 5, 20)
    assert edge_valid.tolist() == [[True, True, True, True, False]]
    assert torch.allclose(features[0, 0, 4:6], torch.tensor([0.5, 0.0]))

    marked, _ = boundary_edge_features(
        points,
        valid,
        entrance_points=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
        entrance_valid=torch.tensor([True]),
    )
    assert marked[0, :, -1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_boundary_edge_encoder_handles_padded_vertices() -> None:
    encoder = BoundaryEdgeGraphEncoder(hidden_dim=16)
    points = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ]
    )
    valid = torch.tensor([[True, True, True, True, False, False]])
    output = encoder(points, valid)
    assert output["tokens"].shape == (1, 6, 16)
    assert output["valid"].sum() == 4
    assert torch.isfinite(output["tokens"]).all()
