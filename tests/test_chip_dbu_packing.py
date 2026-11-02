from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.dbu_packing import repair_dbu_overlaps


def _instance(
    boxes: np.ndarray,
    constraints: np.ndarray,
    *,
    outline: tuple[float, float] | None = None,
) -> FloorplanInstance:
    count = len(boxes)
    targets = np.full((count, 4), -1.0, dtype=np.float64)
    anchored = (constraints[:, 0] != 0) | (constraints[:, 1] != 0)
    targets[anchored] = boxes[anchored]
    return FloorplanInstance(
        block_count=count,
        area_targets=boxes[:, 2] * boxes[:, 3],
        constraints=constraints,
        b2b=np.empty((0, 3), dtype=np.float64),
        p2b=np.empty((0, 3), dtype=np.float64),
        pins_pos=np.empty((0, 2), dtype=np.float64),
        target_positions=targets,
        gt_positions=None,
        outline=outline,
    )


def test_already_feasible_layout_only_snaps_movable_positions() -> None:
    boxes = np.asarray(
        [
            [0.0004, 0.0, 2.0, 2.0],
            [3.0003, 0.0, 1.0, 1.0],
        ]
    )
    constraints = np.zeros((2, 5), dtype=np.float64)
    result = repair_dbu_overlaps(boxes, _instance(boxes, constraints))

    assert result.accepted
    assert result.route == "already_feasible"
    assert result.overlap_after == 0
    assert result.max_grid_residual < 1.0e-9


def test_integer_milp_moves_soft_block_but_preserves_fixed_block() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    constraints = np.zeros((2, 5), dtype=np.float64)
    constraints[0, 0] = 1.0
    result = repair_dbu_overlaps(
        boxes,
        _instance(boxes, constraints, outline=(6.0, 6.0)),
        time_limit_ms=3000,
    )

    assert result.accepted
    assert result.route == "integer_dbu_local_milp"
    assert result.overlap_before == 1
    assert result.overlap_after == 0
    np.testing.assert_allclose(result.boxes[0], boxes[0])
    np.testing.assert_allclose(result.boxes[:, 2:], boxes[:, 2:])
    assert result.moved_blocks == 1


def test_boundary_code_is_enforced_on_integer_grid() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.001, 0.0, 2.0, 2.0],
        ]
    )
    constraints = np.zeros((2, 5), dtype=np.float64)
    constraints[0, 4] = 1.0
    result = repair_dbu_overlaps(
        boxes,
        _instance(boxes, constraints, outline=(5.0, 5.0)),
        max_move_fraction=2.0,
        time_limit_ms=3000,
    )

    assert result.accepted
    assert result.boundary_after == 0
    assert result.boxes[0, 0] == 0.0


def test_fixed_overlap_fails_closed_without_partial_rewrite() -> None:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    constraints = np.zeros((2, 5), dtype=np.float64)
    constraints[:, 0] = 1.0
    result = repair_dbu_overlaps(
        boxes,
        _instance(boxes, constraints, outline=(6.0, 6.0)),
    )

    assert not result.accepted
    assert result.overlap_before == 1
    assert result.overlap_after == 1
    np.testing.assert_array_equal(result.boxes, boxes)
