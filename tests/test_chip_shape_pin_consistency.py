from __future__ import annotations

import torch

from floorflow.chip_flow.shape_pin_consistency import (
    shape_pin_consistency_terms,
)


def _boxes_from_rho(rho: list[float]) -> torch.Tensor:
    ratio = torch.tensor(rho).exp()
    width = ratio.sqrt()
    height = 1.0 / width
    centers = torch.arange(len(rho), dtype=torch.float32)
    return torch.stack(
        [centers, torch.ones_like(centers), width, height],
        dim=-1,
    )[None]


def _empty_edges() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty(1, 0, 3), torch.empty(1, 0, 2), torch.zeros(1)


def test_mib_shared_shape_penalty_requires_common_rho() -> None:
    constraints = torch.zeros(1, 3, 5)
    constraints[..., 2] = 1
    p2b, pins, n_p2b = _empty_edges()
    common = _boxes_from_rho([0.2, 0.2, 0.2])
    divergent = _boxes_from_rho([-0.3, 0.0, 0.4])
    kwargs = {
        "areas": torch.ones(1, 3),
        "constraints": constraints,
        "p2b": p2b,
        "pins": pins,
        "n_p2b": n_p2b,
        "mib_weight": 1.0,
        "pin_tail_weight": 0.0,
        "rho_trust_weight": 0.0,
    }
    shared = shape_pin_consistency_terms(common, common, **kwargs)
    different = shape_pin_consistency_terms(divergent, divergent, **kwargs)
    assert shared.mib_shared < 1.0e-12
    assert different.mib_shared > shared.mib_shared


def test_mib_shared_shape_penalty_rejects_equal_aspect_different_sizes() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 1.0, 2.0], [3.0, 0.0, 2.0, 4.0]]]
    )
    constraints = torch.zeros(1, 2, 5)
    constraints[..., 2] = 1.0
    p2b, pins, n_p2b = _empty_edges()
    terms = shape_pin_consistency_terms(
        boxes,
        boxes,
        torch.ones(1, 2),
        constraints,
        p2b,
        pins,
        n_p2b,
        mib_weight=1.0,
        pin_tail_weight=0.0,
        rho_trust_weight=0.0,
    )

    assert terms.mib_shared > 0


def test_pin_tail_rewards_near_blocks_and_rho_trust_limits_shape_drift() -> None:
    reference = _boxes_from_rho([0.0])
    near = reference.clone()
    near[..., :2] = 0
    far = reference.clone()
    far[..., :2] += 20.0
    p2b = torch.tensor([[[0.0, 0.0, 1.0]]])
    pins = torch.zeros(1, 1, 2)
    constraints = torch.zeros(1, 1, 5)
    near_terms = shape_pin_consistency_terms(
        near,
        reference,
        torch.ones(1, 1),
        constraints,
        p2b,
        pins,
        torch.ones(1),
        pin_tail_weight=1.0,
        rho_trust_weight=0.0,
    )
    far_terms = shape_pin_consistency_terms(
        far,
        reference,
        torch.ones(1, 1),
        constraints,
        p2b,
        pins,
        torch.ones(1),
        pin_tail_weight=1.0,
        rho_trust_weight=0.0,
    )
    assert near_terms.pin_tail < far_terms.pin_tail

    changed = _boxes_from_rho([0.5]).requires_grad_(True)
    within = shape_pin_consistency_terms(
        changed,
        reference,
        torch.ones(1, 1),
        constraints,
        p2b,
        pins,
        torch.zeros(1),
        rho_trust_weight=1.0,
        rho_trust_radius=0.6,
    )
    outside = shape_pin_consistency_terms(
        changed,
        reference,
        torch.ones(1, 1),
        constraints,
        p2b,
        pins,
        torch.zeros(1),
        rho_trust_weight=1.0,
        rho_trust_radius=0.1,
    )
    assert within.rho_trust == 0
    assert outside.rho_trust > 0
    outside.total.backward()
    assert changed.grad is not None
