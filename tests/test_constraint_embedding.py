"""Tests for graph constraint pretraining components."""

from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.constraint_state_encoder import (
    ConstraintStateEncoder,
    d4_transform_graph,
    permute_graph_nodes,
    room_family_ids,
)
from floorflow.objectives.constraint_contrastive import (
    energy_ranking_loss,
    multi_label_violation_loss,
    multi_positive_infonce,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter(
                {"Bedroom": 3, "Kitchen": 3, "Bath": 3, "Hall": 3}
            ),
            "furniture": Counter({"Sink": 3}),
            "door": Counter({"Swing Beside": 3}),
            "window": Counter({"Regular": 3}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_constraint_encoder_exposes_embedding_violations_and_energy() -> None:
    model = ConstraintStateEncoder(hidden_dim=32, embedding_dim=16)
    nodes = 4
    output = model(
        torch.randn(2, nodes, 8),
        torch.eye(nodes)[None].repeat(2, 1, 1),
        torch.zeros(2, nodes, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    assert output["embedding"].shape == (2, 16)
    assert output["violation_logits"].shape == (2, 7)
    assert output["energy"].shape == (2,)
    assert output["legal_logits"].shape == (2,)
    assert torch.allclose(
        output["embedding"].norm(dim=-1),
        torch.ones(2),
        atol=1.0e-5,
    )


def test_constraint_encoder_is_in_parameter_budget() -> None:
    model = ConstraintStateEncoder()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameters <= 5_000_000


def test_d4_and_permutation_preserve_graph_semantics() -> None:
    node_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 0.2, 0.7, 0.0, 0.5],
            [1.0, 2.0, 3.0, 4.0, 0.8, 0.3, 0.0, 0.5],
        ]
    )
    adjacency = torch.tensor([[1.0, 0.25], [0.25, 1.0]])
    rotated, rotated_adjacency = d4_transform_graph(
        node_features,
        adjacency,
        1,
    )
    assert torch.allclose(rotated[:, :4], node_features[:, :4])
    assert torch.equal(rotated_adjacency, adjacency)
    permutation = torch.tensor([1, 0])
    permuted, permuted_adjacency = permute_graph_nodes(
        node_features,
        adjacency,
        permutation,
    )
    assert torch.equal(permuted, node_features[permutation])
    assert torch.equal(permuted_adjacency, adjacency)


def test_room_family_mapping_covers_taxonomy() -> None:
    families = room_family_ids(_taxonomy())
    assert families.shape == (len(_taxonomy().room_ids),)
    assert families.min() >= 0
    assert families.max() < 5


def test_contrastive_and_constraint_losses_are_finite() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    positive = torch.tensor(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )
    contrastive = multi_positive_infonce(embeddings, positive)
    violation = multi_label_violation_loss(
        torch.randn(4, 7, requires_grad=True),
        torch.tensor([[1, 0, 0, 0, 0, 0, 0]] * 2 + [[0] * 7] * 2).float(),
    )
    energy = energy_ranking_loss(
        torch.tensor([0.1, 0.2, 1.5, 1.8]),
        torch.tensor([True, True, False, False]),
    )
    assert contrastive.ndim == 0
    assert violation.ndim == 0
    assert energy.ndim == 0
