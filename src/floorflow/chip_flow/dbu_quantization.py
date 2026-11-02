"""Integer-DBU geometry quantization for chip endpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DbuQuantizationOutput:
    """DBU-aligned boxes and representation-level diagnostics."""

    boxes: torch.Tensor
    area_relative_error: torch.Tensor
    area_violation_mask: torch.Tensor
    max_area_relative_error: float
    failing_block_fraction: float
    max_position_grid_residual: float
    max_shape_grid_residual: float
    max_soft_shape_grid_residual: float
    fixed_shape_error: float
    preplaced_max_error: float
    mib_shape_mismatch_count: int


def _shape_candidate(
    area: float,
    width: float,
    height: float,
    resolution: float,
    tolerance: float,
) -> tuple[int, int]:
    target = max(area / resolution**2, 1.0)
    width_guess = max(1, math.floor(width / resolution + 0.5))
    aspect = math.log(max(width, 1.0e-12) / max(height, 1.0e-12))
    radius = max(2, min(64, math.ceil(1.0 / (2.0 * tolerance))))
    best: tuple[tuple[float, float, float], int, int] | None = None
    for delta in range(-radius, radius + 1):
        candidate_width = max(1, width_guess + delta)
        candidate_height = max(
            1,
            math.floor(target / candidate_width + 0.5),
        )
        relative = abs(candidate_width * candidate_height - target) / target
        candidate_aspect = math.log(candidate_width / candidate_height)
        score = (
            float(relative > tolerance),
            float(relative),
            abs(candidate_aspect - aspect),
        )
        if best is None or score < best[0]:
            best = (score, candidate_width, candidate_height)
    assert best is not None
    return best[1], best[2]


def _mib_shared_shape(
    member_areas: list[float],
    member_shapes: list[tuple[int, int]],
    anchored_shapes: list[tuple[int, int]],
    tolerance: float,
    resolution: float,
) -> tuple[int, int]:
    if anchored_shapes:
        return anchored_shapes[0]
    areas = sorted(member_areas)
    shared_area = areas[len(areas) // 2] / resolution**2
    log_aspects = sorted(
        math.log(shape[0] / shape[1])
        for shape in member_shapes
    )
    shared_aspect = log_aspects[len(log_aspects) // 2]
    base_width = max(
        1,
        math.floor(math.sqrt(shared_area * math.exp(shared_aspect)) + 0.5),
    )
    base_height = max(
        1,
        math.floor(shared_area / base_width + 0.5),
    )
    best: tuple[tuple[float, int, int], int, int] | None = None
    for dw in range(-3, 4):
        for dh in range(-3, 4):
            width = max(1, base_width + dw)
            height = max(1, base_height + dh)
            errors = [
                abs(width * height - area) / max(area, 1.0e-12)
                for area in member_areas
            ]
            score = (
                float(max(errors) > tolerance),
                max(errors),
                abs(dw) + abs(dh),
            )
            if best is None or score < best[0]:
                best = (score, width, height)
    assert best is not None
    return best[1], best[2]


def quantize_boxes_to_dbu(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    *,
    resolution: float = 0.001,
    area_tolerance: float = 0.01,
) -> DbuQuantizationOutput:
    """Quantize centers and soft shapes to an integer database grid.

    The function is representation-only: it does not resolve overlaps or
    boundary violations. Fixed-shape and preplaced inputs are preserved, and
    MIB groups receive one shared quantized shape.
    """

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if area_tolerance < 0.0:
        raise ValueError("area_tolerance must be non-negative")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (batch, blocks, 4)")
    batch, count, _ = boxes.shape
    if areas.shape != (batch, count):
        raise ValueError("areas must match boxes")
    if constraints.shape != (batch, count, 5):
        raise ValueError("constraints must have shape (batch, blocks, 5)")
    if not (
        torch.isfinite(boxes).all()
        and torch.isfinite(areas).all()
        and torch.isfinite(constraints).all()
    ):
        raise ValueError("DBU quantization inputs must be finite")

    original = boxes.float()
    target_areas = areas.float().clamp_min(1.0e-12)
    output = original.clone()
    fixed = constraints[..., 0].ne(0)
    preplaced = constraints[..., 1].ne(0)
    anchored = fixed | preplaced
    mib_ids = constraints[..., 2].long()
    member = mib_ids.gt(0)
    quantized_shapes: list[list[tuple[int, int] | None]] = [
        [None] * count for _ in range(batch)
    ]

    for batch_index in range(batch):
        for index in range(count):
            if anchored[batch_index, index]:
                quantized_shapes[batch_index][index] = (
                    math.floor(
                        float(output[batch_index, index, 2]) / resolution + 0.5
                    ),
                    math.floor(
                        float(output[batch_index, index, 3]) / resolution + 0.5
                    ),
                )
                continue
            quantized_shapes[batch_index][index] = _shape_candidate(
                float(target_areas[batch_index, index]),
                float(original[batch_index, index, 2]),
                float(original[batch_index, index, 3]),
                resolution,
                area_tolerance,
            )

    for batch_index in range(batch):
        for group in torch.unique(mib_ids[batch_index][member[batch_index]]):
            members = torch.nonzero(
                mib_ids[batch_index].eq(group),
                as_tuple=False,
            ).flatten()
            member_indices = members.tolist()
            anchored_indices = [
                index for index in member_indices if anchored[batch_index, index]
            ]
            anchored_shapes = [
                (
                    math.floor(
                        float(output[batch_index, index, 2]) / resolution + 0.5
                    ),
                    math.floor(
                        float(output[batch_index, index, 3]) / resolution + 0.5
                    ),
                )
                for index in anchored_indices
            ]
            if anchored_shapes and any(
                shape != anchored_shapes[0] for shape in anchored_shapes[1:]
            ):
                continue
            shared = _mib_shared_shape(
                [float(target_areas[batch_index, index]) for index in member_indices],
                [
                    quantized_shapes[batch_index][index]
                    for index in member_indices
                    if quantized_shapes[batch_index][index] is not None
                ],
                anchored_shapes,
                area_tolerance,
                resolution,
            )
            for index in member_indices:
                if not anchored[batch_index, index]:
                    quantized_shapes[batch_index][index] = shared

    for batch_index in range(batch):
        for index in range(count):
            width, height = quantized_shapes[batch_index][index]
            if not preplaced[batch_index, index]:
                output[batch_index, index, 0] = (
                    torch.round(original[batch_index, index, 0] / resolution)
                    * resolution
                )
                output[batch_index, index, 1] = (
                    torch.round(original[batch_index, index, 1] / resolution)
                    * resolution
                )
            if not anchored[batch_index, index]:
                output[batch_index, index, 2] = width * resolution
                output[batch_index, index, 3] = height * resolution

    soft = ~anchored
    area_error = (
        output[..., 2] * output[..., 3] - target_areas
    ).abs() / target_areas
    area_error = torch.where(soft, area_error, torch.zeros_like(area_error))
    area_violation = soft & area_error.gt(area_tolerance)

    position_residual = (
        output[..., :2]
        - torch.round(output[..., :2] / resolution) * resolution
    ).abs()
    position_residual = torch.where(
        preplaced[..., None],
        torch.zeros_like(position_residual),
        position_residual,
    )
    shape_residual = (
        output[..., 2:]
        - torch.round(output[..., 2:] / resolution) * resolution
    ).abs()
    soft_shape_residual = shape_residual[soft]

    fixed_error = (output[..., 2:] - original[..., 2:]).abs()[fixed]
    preplaced_error = (output - original).abs()[preplaced]
    mib_mismatches = 0
    for batch_index in range(batch):
        for group in torch.unique(mib_ids[batch_index][member[batch_index]]):
            members = torch.nonzero(
                mib_ids[batch_index].eq(group),
                as_tuple=False,
            ).flatten()
            shapes = {
                (
                    round(float(output[batch_index, index, 2]), 9),
                    round(float(output[batch_index, index, 3]), 9),
                )
                for index in members.tolist()
            }
            mib_mismatches += max(len(shapes) - 1, 0)

    return DbuQuantizationOutput(
        boxes=output.to(dtype=boxes.dtype),
        area_relative_error=area_error,
        area_violation_mask=area_violation,
        max_area_relative_error=float(area_error.max()),
        failing_block_fraction=float(area_violation.float().mean()),
        max_position_grid_residual=float(position_residual.max()),
        max_shape_grid_residual=float(shape_residual.max()),
        max_soft_shape_grid_residual=(
            float(soft_shape_residual.max()) if soft_shape_residual.numel() else 0.0
        ),
        fixed_shape_error=float(fixed_error.max()) if fixed_error.numel() else 0.0,
        preplaced_max_error=(
            float(preplaced_error.max()) if preplaced_error.numel() else 0.0
        ),
        mib_shape_mismatch_count=int(mib_mismatches),
    )


__all__ = ["DbuQuantizationOutput", "quantize_boxes_to_dbu"]
