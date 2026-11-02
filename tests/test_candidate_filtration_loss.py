"""Tests for TopoGED-style CAD candidate descriptors."""

import torch

from floorflow.objectives.candidate_filtration_loss import (
    endpoint_adjacency,
    filtration_descriptor,
    graphlet_descriptor_loss,
)


def test_shared_junctions_and_filtration_are_differentiable() -> None:
    endpoints = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.5, 0.0]],
                [[0.5, 0.0], [0.5, 0.5]],
                [[0.5, 0.5], [1.0, 0.5]],
            ]
        ],
        requires_grad=True,
    )
    valid = torch.ones(1, 3, dtype=torch.bool)
    weights = torch.tensor([[0.8, 0.7, 0.4]], requires_grad=True)

    adjacency = endpoint_adjacency(endpoints, valid)
    descriptor = filtration_descriptor(endpoints, weights, valid)
    loss = graphlet_descriptor_loss(descriptor, descriptor.detach())
    loss.backward()

    assert adjacency.shape == (1, 3, 3)
    assert int(adjacency[0].sum()) == 4
    assert descriptor.shape == (1, 8)
    assert not torch.allclose(descriptor[:, :4], descriptor[:, 4:])
    assert descriptor.requires_grad
    assert weights.grad is not None
