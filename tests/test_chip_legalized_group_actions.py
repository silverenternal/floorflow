from __future__ import annotations

import numpy as np
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.legalized_group_actions import (
    FEATURE_NAMES,
    apply_legalized_group_action,
    enumerate_legalized_group_actions,
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


def test_action_enumeration_is_feasible_and_features_are_stable() -> None:
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
    actions = enumerate_legalized_group_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    assert actions
    for action in actions:
        assert action.features.shape == (len(FEATURE_NAMES),)
        assert action.grouping_after < action.grouping_before
        assert action.boundary_after <= action.boundary_before
        assert np.isfinite(action.features).all()
        np.testing.assert_allclose(
            apply_legalized_group_action(boxes, action),
            action.boxes,
        )


def test_protected_components_generate_no_actions() -> None:
    constraints = np.zeros((2, 5), dtype=np.float32)
    constraints[:, 3] = 1
    constraints[1, 4] = 1
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    assert not enumerate_legalized_group_actions(
        boxes,
        constraints,
        _instance(constraints),
    )


def test_net_cap_filters_actions() -> None:
    constraints = np.zeros((2, 5), dtype=np.float32)
    constraints[:, 3] = 1
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [20.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    instance = _instance(constraints, b2b=[[0.0, 1.0, 100.0]])
    actions = enumerate_legalized_group_actions(
        boxes,
        constraints,
        instance,
        max_move_fraction=1.0,
        max_net_increase=0.0,
    )
    assert all(
        action.b2b_length_after <= action.b2b_length_before
        for action in actions
    )


def test_action_features_are_permutation_invariant() -> None:
    constraints = np.zeros((3, 5), dtype=np.float32)
    constraints[:2, 3] = 1
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 1.0, 1.0], [0.0, 3.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    original = enumerate_legalized_group_actions(
        boxes,
        constraints,
        _instance(constraints),
        max_move_fraction=1.0,
    )
    order = np.asarray([2, 1, 0])
    permuted = enumerate_legalized_group_actions(
        boxes[order],
        constraints[order],
        _instance(constraints[order]),
        max_move_fraction=1.0,
    )
    np.testing.assert_allclose(
        np.sort(np.stack([action.features for action in original]), axis=0),
        np.sort(np.stack([action.features for action in permuted]), axis=0),
    )
