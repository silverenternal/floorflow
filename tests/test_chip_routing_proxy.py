"""Tests for the differentiable routing-pressure proxy."""

from __future__ import annotations

import torch

from floorflow.chip_flow.batched import (
    batch_grouping_proxy,
    batch_routing_proxy_loss,
    batch_routing_proxy_terms,
)


def test_routing_proxy_is_finite_differentiable_and_penalizes_hotspots() -> None:
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [20.0, 20.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [5.0, 0.0, 1.0, 1.0],
                [10.0, 0.0, 1.0, 1.0],
                [20.0, 20.0, 1.0, 1.0],
            ],
        ],
        requires_grad=True,
    )
    areas = torch.ones((2, 4))
    b2b = torch.tensor(
        [
            [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]],
            [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]],
        ]
    )
    p2b = torch.tensor(
        [
            [[0, 0, 1.0], [1, 1, 1.0]],
            [[0, 0, 1.0], [1, 1, 1.0]],
        ]
    )
    pins = torch.tensor(
        [
            [[0.5, 0.5], [1.5, 0.5]],
            [[0.5, 0.5], [5.5, 0.5]],
        ]
    )
    terms = batch_routing_proxy_terms(
        boxes,
        areas,
        b2b,
        p2b,
        pins,
        torch.tensor([3, 3]),
        torch.tensor([2, 2]),
        torch.tensor([2, 2]),
    )

    for value in terms.values():
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    assert terms["occupancy_peak"][0] > terms["occupancy_peak"][1]
    sum(terms.values()).sum().backward()
    assert boxes.grad is not None
    assert torch.isfinite(boxes.grad).all()


def test_grouping_proxy_penalizes_large_cluster_gaps() -> None:
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0]],
        ],
        requires_grad=True,
    )
    constraints = torch.zeros((2, 2, 5))
    constraints[..., 3] = 1.0
    values = batch_grouping_proxy(boxes, constraints)

    assert values.shape == (2,)
    assert values[0] < values[1]
    values.sum().backward()
    assert boxes.grad is not None
    assert torch.isfinite(boxes.grad).all()


def test_normalized_routing_loss_has_stable_scale() -> None:
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
                [20.0, 20.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    areas = torch.ones((1, 4))
    b2b = torch.tensor([[[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]]])
    p2b = torch.tensor([[[0, 0, 1.0], [1, 1, 1.0]]])
    pins = torch.tensor([[[0.5, 0.5], [1.5, 0.5]]])
    loss = batch_routing_proxy_loss(
        boxes,
        areas,
        b2b,
        p2b,
        pins,
        torch.tensor([3]),
        torch.tensor([2]),
        torch.tensor([2]),
    )

    assert loss.shape == (1,)
    assert torch.isfinite(loss).all()
    assert 0.0 < float(loss.detach()) < 10.0


def test_b2b_scale_does_not_suppress_the_p2b_route_signal() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 1.0], [9.0, 0.0, 1.0, 1.0]]]
    )
    terms = batch_routing_proxy_terms(
        boxes,
        torch.ones(1, 2),
        torch.tensor([[[0.0, 1.0, 1000.0]]]),
        torch.tensor([[[0.0, 1.0, 0.01]]]),
        torch.tensor([[[0.1, 0.9]]]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
    )

    assert terms["b2b_route_demand_peak"][0] > 0
    assert terms["p2b_route_demand_peak"][0] > 0
    torch.testing.assert_close(
        terms["route_demand_peak"][0],
        0.5
        * (
            terms["b2b_route_demand_peak"][0]
            + terms["p2b_route_demand_peak"][0]
        ),
    )


def test_routing_proxy_is_invariant_to_translation_and_physical_units() -> None:
    boxes = torch.tensor(
        [
            [
                [1.0, 2.0, 2.0, 1.0],
                [4.0, 2.5, 1.0, 2.0],
                [8.0, 5.0, 2.0, 2.0],
                [14.0, 9.0, 1.0, 1.0],
            ]
        ],
        dtype=torch.float64,
    )
    areas = boxes[..., 2] * boxes[..., 3]
    b2b = torch.tensor(
        [[[0.0, 1.0, 1.0], [1.0, 2.0, 2.0], [2.0, 3.0, 1.0]]],
        dtype=torch.float64,
    )
    p2b = torch.tensor(
        [[[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]]],
        dtype=torch.float64,
    )
    pins = torch.tensor(
        [[[0.0, 0.0], [12.0, 3.0]]],
        dtype=torch.float64,
    )
    counts = (
        torch.tensor([3]),
        torch.tensor([2]),
        torch.tensor([2]),
    )
    baseline = batch_routing_proxy_terms(
        boxes,
        areas,
        b2b,
        p2b,
        pins,
        *counts,
    )

    shift = torch.tensor([17.0, -9.0], dtype=torch.float64)
    scale = 1000.0
    moved_boxes = (boxes + torch.cat([shift, shift.new_zeros(2)])) * scale
    moved_areas = areas * scale**2
    moved_pins = (pins + shift) * scale
    moved = batch_routing_proxy_terms(
        moved_boxes,
        moved_areas,
        b2b,
        p2b,
        moved_pins,
        *counts,
    )

    for name, value in baseline.items():
        torch.testing.assert_close(moved[name], value, rtol=1.0e-10, atol=1.0e-12)


def test_route_obstruction_penalizes_macro_path_blockage() -> None:
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [4.0, 0.0, 1.0, 1.0],
                [8.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [4.0, 0.0, 1.0, 1.0],
                [8.0, 0.0, 1.0, 1.0],
                [2.0, 5.0, 1.0, 1.0],
            ],
        ]
    )
    terms = batch_routing_proxy_terms(
        boxes,
        torch.ones(2, 4),
        torch.tensor([[[0.0, 1.0, 1.0]]] * 2),
        torch.empty(2, 0, 3),
        torch.empty(2, 0, 2),
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )

    assert terms["route_obstruction"][0] > terms["route_obstruction"][1]
