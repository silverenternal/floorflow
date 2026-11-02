from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_chip_dbu_packing_gate import (
    _case_indices,
    _instance,
    _strict_gate,
)


def test_case_indices_preserve_order_and_apply_limit() -> None:
    assert _case_indices("57, 231,57,254", 0) == [57, 231, 254]
    assert _case_indices("57,231,254", 2) == [57, 231]


def test_original_instance_keeps_fixed_and_preplaced_targets() -> None:
    geometry = {
        "areas": np.asarray([4.0, 6.0, 8.0]),
        "constraints": np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "gt_boxes": np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 2.0, 3.0],
                [5.0, 0.0, 2.0, 4.0],
            ]
        ),
        "b2b": np.empty((0, 3)),
        "p2b": np.empty((0, 3)),
        "pins": np.empty((0, 2)),
    }
    instance = _instance(geometry)

    np.testing.assert_array_equal(instance.target_positions[0], [-1.0] * 4)
    np.testing.assert_array_equal(instance.target_positions[1, 2:], [2.0, 3.0])
    np.testing.assert_array_equal(instance.target_positions[2], [5.0, 0.0, 2.0, 4.0])


def test_strict_gate_requires_reference_soft_metrics() -> None:
    reference = {"grouping": 1, "boundary": 2}
    repaired = {
        "hard_feasible": True,
        "area_violations": 0,
        "fixed_violations": 0,
        "preplaced_violations": 0,
        "aspect_violations": 0,
        "max_grid_residual": 0.0,
        "grouping": 1,
        "boundary": 2,
    }
    gate = _strict_gate(
        reference,
        repaired,
        accepted=True,
        max_displacement=0.5,
        span=10.0,
        grid_tolerance=1.0e-4,
        max_displacement_fraction=0.5,
    )

    assert gate["passed"]
    repaired["boundary"] = 3
    gate = _strict_gate(
        reference,
        repaired,
        accepted=True,
        max_displacement=0.5,
        span=10.0,
        grid_tolerance=1.0e-4,
        max_displacement_fraction=0.5,
    )
    assert not gate["passed"]
