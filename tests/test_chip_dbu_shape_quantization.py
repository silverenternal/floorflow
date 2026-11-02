from __future__ import annotations

import numpy as np
import torch
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.dbu_shape_quantization import (
    quantize_boxes_adjacency_aware,
)


def _inputs():
    boxes = np.asarray(
        [
            [0.0004, 0.0004, 2.0004, 2.0003],
            [2.0008, 0.0004, 2.0002, 2.0004],
            [3.5002, 2.0007, 2.5003, 2.0004],
            [0.5002, 4.0003, 1.5002, 1.5004],
            [0.0, 6.0003, 2.0003, 1.0004],
        ],
        dtype=np.float64,
    )
    constraints = np.zeros((len(boxes), 5), dtype=np.float64)
    constraints[0, 3] = 1.0
    constraints[1, 3] = 1.0
    constraints[2, 4] = 2.0
    constraints[3, 1] = 1.0
    constraints[4, 0] = 1.0
    constraints[4, 4] = 1.0
    targets = np.full((len(boxes), 4), -1.0, dtype=np.float64)
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    targets[fixed, 2:] = boxes[fixed, 2:]
    targets[preplaced] = boxes[preplaced]
    instance = FloorplanInstance(
        block_count=len(boxes),
        area_targets=boxes[:, 2] * boxes[:, 3],
        constraints=constraints,
        b2b=np.empty((0, 3), dtype=np.float64),
        p2b=np.empty((0, 3), dtype=np.float64),
        pins_pos=np.empty((0, 2), dtype=np.float64),
        target_positions=targets,
        gt_positions=boxes,
    )
    return boxes, constraints, instance


def test_shape_quantization_preserves_soft_and_anchor_contracts() -> None:
    boxes, constraints, instance = _inputs()
    result = quantize_boxes_adjacency_aware(
        torch.from_numpy(boxes.astype(np.float32))[None],
        torch.from_numpy((boxes[:, 2] * boxes[:, 3]).astype(np.float32))[None],
        torch.from_numpy(constraints.astype(np.float32))[None],
        instance,
    )

    assert result.max_grid_residual <= 1.0e-4
    assert not result.area_violation_mask.any()
    assert result.fixed_shape_error == 0.0
    assert result.preplaced_max_error == 0.0
    assert result.grouping_after <= result.grouping_before
    assert result.boundary_after <= result.boundary_before


def test_shape_quantization_rejects_batch_greater_than_one() -> None:
    boxes, constraints, instance = _inputs()
    try:
        quantize_boxes_adjacency_aware(
            torch.from_numpy(boxes.astype(np.float32))[None].repeat(2, 1, 1),
            torch.from_numpy((boxes[:, 2] * boxes[:, 3]).astype(np.float32))[None],
            torch.from_numpy(constraints.astype(np.float32))[None],
            instance,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("batch greater than one must fail")
