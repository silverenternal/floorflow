from __future__ import annotations

import torch

from floorflow.chip_flow.boundary_snap import (
    boundary_violations,
    snap_boundary_boxes,
)


def test_boundary_snap_repairs_movable_coded_block() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
            [0.0, 3.0, 1.0, 1.0],
        ]
    )
    codes = torch.tensor([1.0, 4.0, 0.0])
    result = snap_boundary_boxes(
        boxes,
        codes,
        torch.tensor([False, True, False]),
    )
    assert result.before == 1
    assert result.after == 0
    assert result.moves == 1
    torch.testing.assert_close(result.boxes[1], torch.tensor([2.0, 3.0, 1.0, 1.0]))


def test_boundary_snap_does_not_create_overlap() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.1, 0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 1.0],
        ]
    )
    codes = torch.tensor([0.0, 1.0, 0.0])
    result = snap_boundary_boxes(
        boxes,
        codes,
        torch.tensor([False, True, False]),
    )
    assert result.before == 1
    assert result.after == 0
    assert result.moves == 1
    assert boundary_violations(result.boxes, codes) == 0
    assert result.boxes[0, 2] == 1.0
