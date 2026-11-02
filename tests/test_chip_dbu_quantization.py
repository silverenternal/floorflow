from __future__ import annotations

import torch

from floorflow.chip_flow.dbu_quantization import quantize_boxes_to_dbu


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = torch.tensor(
        [
            [
                [0.0004, 0.0004, 2.0004, 3.0007],
                [2.0002, 0.0004, 1.5002, 4.0008],
                [5.0004, 1.0003, 2.0002, 2.0004],
                [8.0001, 2.0001, 2.0003, 2.0005],
            ]
        ],
        dtype=torch.float32,
    )
    areas = boxes[..., 2] * boxes[..., 3]
    constraints = torch.zeros(1, 4, 5)
    constraints[0, 2, 0] = 1
    constraints[0, 3, 1] = 1
    constraints[0, 0, 2] = 1
    constraints[0, 1, 2] = 1
    return boxes, areas, constraints


def test_soft_blocks_are_dbu_aligned_and_area_safe() -> None:
    boxes, areas, constraints = _inputs()
    result = quantize_boxes_to_dbu(
        boxes,
        areas,
        constraints,
        resolution=0.001,
        area_tolerance=0.01,
    )
    assert result.max_position_grid_residual < 1.0e-6
    assert result.max_soft_shape_grid_residual < 1.0e-6
    assert result.max_area_relative_error <= 0.01
    assert result.failing_block_fraction == 0.0


def test_fixed_and_preplaced_inputs_are_preserved() -> None:
    boxes, areas, constraints = _inputs()
    result = quantize_boxes_to_dbu(boxes, areas, constraints)
    torch.testing.assert_close(result.boxes[0, 2, 2:], boxes[0, 2, 2:])
    torch.testing.assert_close(result.boxes[0, 3], boxes[0, 3])
    assert result.fixed_shape_error == 0.0
    assert result.preplaced_max_error == 0.0


def test_mib_group_receives_one_grid_shape() -> None:
    boxes, areas, constraints = _inputs()
    result = quantize_boxes_to_dbu(boxes, areas, constraints)
    assert result.mib_shape_mismatch_count == 0
    torch.testing.assert_close(result.boxes[0, 0, 2:], result.boxes[0, 1, 2:])


def test_quantization_is_block_permutation_equivariant() -> None:
    boxes, areas, constraints = _inputs()
    original = quantize_boxes_to_dbu(boxes, areas, constraints)
    order = torch.tensor([3, 1, 0, 2])
    permuted = quantize_boxes_to_dbu(
        boxes[:, order],
        areas[:, order],
        constraints[:, order],
    )
    torch.testing.assert_close(permuted.boxes, original.boxes[:, order])
    torch.testing.assert_close(
        permuted.area_relative_error,
        original.area_relative_error[:, order],
    )


def test_invalid_resolution_fails_closed() -> None:
    boxes, areas, constraints = _inputs()
    try:
        quantize_boxes_to_dbu(boxes, areas, constraints, resolution=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero DBU resolution must fail")
