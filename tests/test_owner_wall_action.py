"""Tests for topology-constrained owner-wall action selection."""

import numpy as np

from floorflow.inference.owner_wall_action import (
    select_connected_wall_actions,
)


def test_wall_action_selection_keeps_a_spanning_forest() -> None:
    endpoints = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [3.0, 0.0]],
        ]
    )
    scores = np.asarray([0.9, 0.1, 0.1, 0.8])
    valid = np.ones(4, dtype=bool)

    selected = select_connected_wall_actions(
        endpoints,
        scores,
        valid,
        np.zeros(4, dtype=bool),
        threshold=0.7,
        coordinate_scale=100.0,
        cluster_tolerance=0.1,
    )

    assert selected.tolist() == [True, True, False, True]


def test_wall_action_selection_forces_exterior_and_ignores_invalid() -> None:
    endpoints = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ]
    )
    selected = select_connected_wall_actions(
        endpoints,
        np.asarray([0.1, 0.1]),
        np.asarray([True, False]),
        np.asarray([True, False]),
        threshold=0.5,
        coordinate_scale=100.0,
        cluster_tolerance=0.1,
    )

    assert selected.tolist() == [True, False]
