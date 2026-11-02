from __future__ import annotations

import torch

from floorflow.chip_flow.legal_parent_reflow import critical_block_mask


def test_critical_block_mask_excludes_fixed_and_preplaced_blocks() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 1.0],
        ]
    )
    b2b = torch.tensor([[0.0, 2.0, 1.0], [2.0, 3.0, 5.0]])
    p2b = torch.tensor([[0.0, 3.0, 10.0]])
    pins = torch.tensor([[8.0, 0.0]])
    constraints = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    mask = critical_block_mask(
        boxes,
        b2b,
        p2b,
        pins,
        constraints,
        2,
        1,
        top_fraction=0.5,
    )
    assert not bool(mask[0])
    assert not bool(mask[1])
    assert bool(mask[2])
    assert bool(mask[3])


def test_critical_block_mask_selects_at_least_one_movable_block() -> None:
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    constraints = torch.zeros(2, 5)
    mask = critical_block_mask(
        boxes,
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0, 2),
        constraints,
        0,
        0,
        top_fraction=0.01,
    )
    assert int(mask.sum()) == 1
