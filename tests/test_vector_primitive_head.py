"""Tests for query-based vector primitive decoding."""

from __future__ import annotations

import torch

from floorflow.heads.vector_primitive_head import VectorPrimitiveHead


def _tensor_values(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _tensor_values(nested)


def test_vector_primitive_head_shapes_and_gradients() -> None:
    head = VectorPrimitiveHead(
        in_channels=16,
        room_classes=5,
        opening_classes=3,
        hidden_dim=32,
        max_rooms=7,
        max_walls=9,
        max_openings=6,
        polygon_points=8,
        decoder_layers=1,
        stage_context=True,
        host_conditioned_openings=True,
    )
    features = torch.randn(2, 16, 16, 16, requires_grad=True)
    graph = torch.randn(2, 4, 32)
    graph_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]],
        dtype=torch.bool,
    )

    output = head(features, graph, graph_mask)
    loss = sum(
        value.float().square().mean()
        for value in _tensor_values(output)
    )
    loss.backward()

    assert output["room_polygons"].shape == (2, 7, 8, 2)
    assert output["wall_endpoints"].shape == (2, 9, 2, 2)
    assert output["opening_centers"].shape == (2, 6, 2)
    assert output["opening_host_logits"].shape == (2, 6, 9)
    assert features.grad is not None
    assert head.room_polygon.weight.grad is not None


def test_partition_edge_condition_reaches_wall_decoder() -> None:
    head = VectorPrimitiveHead(
        in_channels=8,
        room_classes=4,
        opening_classes=2,
        hidden_dim=16,
        max_rooms=4,
        max_walls=8,
        max_openings=2,
        polygon_points=6,
        decoder_layers=1,
        partition_edge_condition=True,
    )
    features = torch.randn(1, 8, 16, 16)
    edge = torch.zeros(1, 16, 16)
    edge[:, :, 8] = 1.0
    output = head(features, partition_edge=edge)
    sum(
        value.float().square().mean()
        for value in _tensor_values(output)
    ).backward()

    assert output["wall_endpoints"].shape == (1, 8, 2, 2)
    assert head.partition_edge_projection is not None
    assert head.partition_edge_projection.weight.grad is not None
    assert torch.count_nonzero(head.partition_edge_projection.weight.grad) > 0
