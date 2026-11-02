from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.legalized_group_repair import repair_legalized_groups


def _instance(constraints: np.ndarray) -> FloorplanInstance:
    count = len(constraints)
    targets = np.full((count, 4), -1.0, dtype=np.float32)
    return FloorplanInstance(
        block_count=count,
        area_targets=np.ones(count, dtype=np.float32),
        constraints=constraints,
        b2b=np.empty((0, 3), dtype=np.float32),
        p2b=np.empty((0, 3), dtype=np.float32),
        pins_pos=np.empty((0, 2), dtype=np.float32),
        target_positions=targets,
        gt_positions=np.zeros((count, 4), dtype=np.float32),
    )


def test_group_repair_attaches_split_component_without_boundary_loss() -> None:
    constraints = np.zeros((3, 5), dtype=np.float32)
    constraints[:2, 3] = 1
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [0.0, 3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = repair_legalized_groups(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    assert result.feasible
    assert result.grouping_after < result.grouping_before
    assert result.boundary_after <= result.boundary_before
    assert result.moves > 0


def test_group_repair_never_moves_fixed_or_boundary_components() -> None:
    constraints = np.zeros((2, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[1, 0] = 1
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    result = repair_legalized_groups(boxes, constraints, _instance(constraints))
    np.testing.assert_array_equal(result.boxes, boxes)
    assert result.moves == 0
