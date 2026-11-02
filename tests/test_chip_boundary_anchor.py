from __future__ import annotations

import torch

from floorflow.chip_flow.boundary_anchor_coupler import BoundaryAnchorCoupler
from floorflow.chip_flow.hierarchical_flow import (
    HierarchicalTokenFlowModel,
    build_hierarchy_targets,
)
from floorflow.chip_flow.model import encode_latents


def _batch() -> dict:
    count = 6
    boxes = torch.zeros(2, count, 4)
    boxes[..., 2:] = 1.0
    boxes[..., 0] = torch.arange(count) * 1.2
    constraints = torch.zeros(2, count, 5)
    constraints[:, :3, 4] = 1.0
    constraints[:, 0, 1] = 1.0
    constraints[:, 1, 0] = 1.0
    return {
        "areas": torch.ones(2, count),
        "boxes": boxes,
        "constraints": constraints,
        "features": torch.zeros(2, count, 10),
        "target_latent": torch.randn(2, count, 3),
        "b2b": torch.tensor(
            [
                [[0.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
                [[0.0, 2.0, 1.0], [2.0, 4.0, 1.0]],
            ]
        ),
        "p2b": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 2.0, 1.0], [1.0, 4.0, 1.0]],
            ]
        ),
        "pins": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ]
        ),
        "n_b2b": torch.tensor([2, 2]),
        "n_p2b": torch.tensor([2, 2]),
        "n_pins": torch.tensor([2, 2]),
    }


def test_boundary_anchor_is_zero_initialized() -> None:
    torch.manual_seed(7)
    base = HierarchicalTokenFlowModel(hidden=64, heads=4, depth=1)
    anchored = HierarchicalTokenFlowModel(
        hidden=64,
        heads=4,
        depth=1,
        boundary_anchor_conditioning=True,
    )
    anchored.load_state_dict(base.state_dict(), strict=False)
    batch = _batch()
    base_context = base.prepare_context(batch)
    anchored_context = anchored.prepare_context(batch)
    target = build_hierarchy_targets(batch, base_context)
    time = torch.full((2,), 0.5)

    expected = base(target, time, base_context).block_state
    actual = anchored(target, time, anchored_context).block_state

    torch.testing.assert_close(actual, expected)
    assert anchored.boundary_anchor is not None


def test_boundary_anchor_accepts_only_boundary_improving_moves() -> None:
    torch.manual_seed(11)
    module = BoundaryAnchorCoupler(16, bottleneck=8)
    module.gate.data.fill_(1.0)
    torch.nn.init.zeros_(module.encoder[-1].weight)
    module.encoder[-1].bias.data.copy_(torch.tensor([-0.1, 0.0, 0.0]))
    boxes = torch.tensor(
        [[[1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]]]
    )
    areas = torch.ones(1, 3)
    state = encode_latents(boxes, areas)
    boundary = torch.zeros(1, 3, 4, dtype=torch.bool)
    boundary[0, 0, 0] = True
    constraints = torch.zeros(1, 3, 5)
    residual, diagnostics = module(
        torch.zeros(1, 3, 16),
        state,
        boundary,
        torch.tensor([0.5]),
        areas=areas,
        constraints=constraints,
    )

    assert not torch.equal(residual[0, 0], torch.zeros(3))
    torch.testing.assert_close(residual[0, 1:], torch.zeros(2, 3))
    assert diagnostics["boundary_anchor_accept_fraction"] == 1.0 / 3.0

    constraints[0, 0, 1] = 1.0
    locked, _ = module(
        torch.zeros(1, 3, 16),
        state,
        boundary,
        torch.tensor([0.5]),
        areas=areas,
        constraints=constraints,
    )
    torch.testing.assert_close(locked, torch.zeros_like(locked))
