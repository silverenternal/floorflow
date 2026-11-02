"""Directional tests for soft graph edit distance."""

from __future__ import annotations

import torch

from floorflow.objectives.soft_graph_edit import soft_graph_edit_distance


def _inputs() -> tuple[torch.Tensor, ...]:
    tokens = torch.tensor(
        [
            [
                [0.2, 0.2, 0.2, 0.0],
                [0.2, 0.8, 0.2, 0.0],
                [0.2, 0.5, 0.8, 0.0],
            ]
        ],
        requires_grad=True,
    )
    classes = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    adjacency = torch.tensor(
        [
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    return tokens, classes, adjacency


def test_identical_graph_has_lower_soft_edit_than_substitution() -> None:
    tokens, classes, adjacency = _inputs()
    identical, _ = soft_graph_edit_distance(
        tokens,
        tokens.detach(),
        classes,
        classes.detach(),
        adjacency,
        adjacency.detach(),
    )
    changed_classes = classes.detach().clone()
    changed_classes[:, 2] = torch.tensor([0.0, 1.0])
    substituted, _ = soft_graph_edit_distance(
        tokens,
        tokens.detach(),
        classes,
        changed_classes,
        adjacency,
        adjacency.detach(),
        aligned_node_order=False,
        aligned_edge_order=False,
    )
    assert substituted > identical


def test_edge_deletion_increases_soft_edge_loss() -> None:
    tokens, classes, adjacency = _inputs()
    deleted = adjacency.detach().clone()
    deleted[:, 0, 1] = 0.0
    deleted[:, 1, 0] = 0.0
    intact, intact_terms = soft_graph_edit_distance(
        tokens,
        tokens.detach(),
        classes,
        classes.detach(),
        adjacency,
        adjacency.detach(),
    )
    removed, removed_terms = soft_graph_edit_distance(
        tokens,
        tokens.detach(),
        classes,
        classes.detach(),
        adjacency,
        deleted,
    )
    assert removed_terms["soft_edge_edit"] > intact_terms["soft_edge_edit"]
    assert removed > intact
    intact.backward()
    assert tokens.grad is not None
