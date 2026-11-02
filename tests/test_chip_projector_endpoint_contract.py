from __future__ import annotations

import torch

from floorflow.chip_flow.projector_endpoint_contract import (
    projector_aware_endpoint,
)


def _inputs():
    source = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.2, 0.1, 0.1]]],
        requires_grad=True,
    )
    target = source.detach().clone()
    constraints = torch.zeros(1, 3, 5)
    return source, target, constraints


def test_straight_through_endpoint_has_projected_value_and_identity_gradient() -> None:
    source, target, constraints = _inputs()
    areas = torch.ones(1, 3)
    projected, terms = projector_aware_endpoint(
        source,
        target,
        constraints,
        areas,
        iterations=4,
        straight_through=False,
        strength=1.0,
    )
    straight, straight_terms = projector_aware_endpoint(
        source,
        target,
        constraints,
        areas,
        iterations=4,
        straight_through=True,
        strength=1.0,
    )
    torch.testing.assert_close(straight, projected)
    assert terms.keys() == straight_terms.keys()
    straight.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


def test_endpoint_contract_does_not_read_free_target_positions() -> None:
    source, target, constraints = _inputs()
    areas = torch.ones(1, 3)
    first, _ = projector_aware_endpoint(
        source,
        target,
        constraints,
        areas,
        iterations=4,
        straight_through=False,
        strength=1.0,
    )
    changed = target.clone()
    changed[..., :2] += 10.0
    second, _ = projector_aware_endpoint(
        source,
        changed,
        constraints,
        areas,
        iterations=4,
        straight_through=False,
        strength=1.0,
    )
    assert torch.equal(first, second)


def test_zero_strength_preserves_raw_endpoint_exactly() -> None:
    source, target, constraints = _inputs()
    endpoint, _ = projector_aware_endpoint(
        source,
        target,
        constraints,
        torch.ones(1, 3),
        iterations=4,
        straight_through=True,
        strength=0.0,
    )
    assert torch.equal(endpoint, source)
