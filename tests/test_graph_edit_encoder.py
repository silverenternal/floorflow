"""Tests for graph-edit conditioning."""

from __future__ import annotations

import torch

from floorflow.model.graph_edit_encoder import GraphEditEncoder


def test_graph_edit_encoder_shapes_and_masking() -> None:
    encoder = GraphEditEncoder(node_features=7, hidden_dim=32, layers=2)
    features = torch.randn(2, 5, 7)
    adjacency = torch.eye(5).expand(2, -1, -1)
    family = torch.zeros(2, 5, dtype=torch.long)
    operation = torch.tensor([1, 3])
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )

    output = encoder(features, adjacency, family, operation, mask)

    assert output["node_tokens"].shape == (2, 5, 32)
    assert output["global_token"].shape == (2, 32)
    assert torch.equal(output["node_mask"], mask)
    assert torch.count_nonzero(output["node_tokens"][0, 3:]) == 0
