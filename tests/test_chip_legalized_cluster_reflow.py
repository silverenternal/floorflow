from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.legalized_cluster_reflow import (
    CLUSTER_REFLOW_FEATURE_NAMES,
    apply_cluster_reflow_action,
    enumerate_cluster_reflow_actions,
    select_cluster_reflow_action,
)


def _instance(
    constraints: np.ndarray,
    *,
    b2b: np.ndarray | None = None,
) -> FloorplanInstance:
    count = len(constraints)
    return FloorplanInstance(
        block_count=count,
        area_targets=np.ones(count, dtype=np.float32),
        constraints=constraints,
        b2b=(
            np.empty((0, 3), dtype=np.float32)
            if b2b is None
            else np.asarray(b2b, dtype=np.float32)
        ),
        p2b=np.empty((0, 3), dtype=np.float32),
        pins_pos=np.empty((0, 2), dtype=np.float32),
        target_positions=np.full((count, 4), -1.0, dtype=np.float32),
        gt_positions=np.zeros((count, 4), dtype=np.float32),
    )


def test_cluster_reflow_builds_composite_action_and_strictly_reduces_split() -> None:
    constraints = np.zeros((3, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[0, 0] = 1
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [3.0, 3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    actions = enumerate_cluster_reflow_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    assert actions
    action = actions[0]
    assert action.grouping_after < action.grouping_before
    assert action.moves
    assert all(index != 0 for index, _dx, _dy in action.moves)
    np.testing.assert_allclose(action.boxes[0], boxes[0])
    np.testing.assert_allclose(
        apply_cluster_reflow_action(boxes, action),
        action.boxes,
    )
    assert action.features.shape == (len(CLUSTER_REFLOW_FEATURE_NAMES),)
    assert np.isfinite(action.features).all()


def test_cluster_reflow_actions_are_quantized_before_acceptance() -> None:
    constraints = np.zeros((3, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[0, 0] = 1
    boxes = np.asarray(
        [
            [0.0004, 0.0, 1.0, 1.0],
            [3.0004, 0.0, 1.0, 1.0],
            [3.0004, 3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    actions = enumerate_cluster_reflow_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
        coordinate_resolution_um=0.001,
    )
    assert actions
    for action in actions:
        moved = [index for index, _dx, _dy in action.moves]
        np.testing.assert_allclose(
            action.boxes[moved, :2],
            np.round(action.boxes[moved, :2] / 0.001) * 0.001,
        )
        replay = apply_cluster_reflow_action(boxes, action)
        np.testing.assert_allclose(replay, action.boxes)


def test_cluster_reflow_respects_net_cap() -> None:
    constraints = np.zeros((2, 5), dtype=np.float32)
    constraints[:, 3] = 1
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [20.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    actions = enumerate_cluster_reflow_actions(
        boxes,
        constraints,
        _instance(constraints, b2b=[[0.0, 1.0, 100.0]]),
        max_move_fraction=1.0,
        max_net_increase=0.0,
    )
    assert all(
        action.b2b_length_after <= action.b2b_length_before
        for action in actions
    )


def test_cluster_reflow_is_permutation_invariant() -> None:
    constraints = np.zeros((3, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[0, 0] = 1
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [3.0, 3.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    original = enumerate_cluster_reflow_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    order = np.asarray([2, 0, 1])
    permuted = enumerate_cluster_reflow_actions(
        boxes[order],
        constraints[order],
        _instance(constraints[order]),
        max_move_fraction=1.0,
    )
    np.testing.assert_allclose(
        np.sort(np.stack([action.features for action in original]), axis=0),
        np.sort(np.stack([action.features for action in permuted]), axis=0),
    )


def test_cluster_reflow_policy_selects_net_preserving_action() -> None:
    constraints = np.zeros((4, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[0, 0] = 1
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [3.0, 3.0, 1.0, 1.0],
            [6.0, 6.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    actions = enumerate_cluster_reflow_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    selected = select_cluster_reflow_action(actions)
    assert selected is not None
    assert selected in actions
