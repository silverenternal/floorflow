"""Fail-closed local repair of quantized chip placements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.soft import check_soft
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.outline_packing import pack_fixed_outline


@dataclass(frozen=True)
class DbuPackingResult:
    """A repaired placement and its exact post-quantization audit."""

    boxes: np.ndarray
    accepted: bool
    route: str
    overlap_before: int
    overlap_after: int
    area_violations: int
    fixed_violations: int
    preplaced_violations: int
    grouping_before: int
    grouping_after: int
    boundary_before: int
    boundary_after: int
    max_grid_residual: float
    max_displacement: float
    moved_blocks: int
    solver_calls: int


def _audit(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    resolution: float,
) -> tuple:
    report = check_feasibility(Placement(boxes, instance))
    soft = check_soft(Placement(boxes, instance))
    aligned = np.rint(boxes / resolution) * resolution
    return (
        report,
        soft,
        float(np.max(np.abs(boxes - aligned), initial=0.0)),
    )


def _overlap_components(
    count: int,
    pairs: list[tuple[int, int]],
) -> list[list[int]]:
    adjacency = [set() for _ in range(count)]
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[list[int]] = []
    seen: set[int] = set()
    for start in range(count):
        if start in seen or not adjacency[start]:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in sorted(adjacency[index]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        result.append(sorted(component))
    return result


def _intersects(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(
        min(float(left[0] + left[2]), float(right[0] + right[2]))
        >= max(float(left[0]), float(right[0])) - tolerance
        and min(float(left[1] + left[3]), float(right[1] + right[3]))
        >= max(float(left[1]), float(right[1])) - tolerance
    )


def _local_region(
    boxes: np.ndarray,
    seed: list[int],
    instance: FloorplanInstance,
    *,
    expansion: float,
    tolerance: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    seed_box = boxes[seed]
    x0 = float(seed_box[:, 0].min())
    y0 = float(seed_box[:, 1].min())
    x1 = float((seed_box[:, 0] + seed_box[:, 2]).max())
    y1 = float((seed_box[:, 1] + seed_box[:, 3]).max())
    span = max(x1 - x0, y1 - y0, float(boxes[:, 2:].max()), 1.0)
    pad = expansion * span
    region = np.asarray(
        [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
        dtype=np.float64,
    )
    if instance.outline is not None:
        region[0] = max(0.0, region[0])
        region[1] = max(0.0, region[1])
        region[2] = min(float(instance.outline[0]), region[2])
        region[3] = min(float(instance.outline[1]), region[3])

    selected: set[int] = set(seed)
    changed = True
    while changed:
        changed = False
        region_xywh = np.asarray(
            [region[0], region[1], region[2] - region[0], region[3] - region[1]]
        )
        for index, box in enumerate(boxes):
            if index in selected or not _intersects(box, region_xywh, tolerance):
                continue
            selected.add(index)
            region[0] = min(region[0], float(box[0]))
            region[1] = min(region[1], float(box[1]))
            region[2] = max(region[2], float(box[0] + box[2]))
            region[3] = max(region[3], float(box[1] + box[3]))
            if instance.outline is not None:
                region[0] = max(0.0, region[0])
                region[1] = max(0.0, region[1])
                region[2] = min(float(instance.outline[0]), region[2])
                region[3] = min(float(instance.outline[1]), region[3])
            changed = True

    # A boundary-coded block needs the global outline edge, not a local cut edge.
    codes = instance.boundary_code
    selected_codes = codes[list(selected)]
    if np.any(selected_codes):
        if instance.outline is not None:
            region[:] = [0.0, 0.0, instance.outline[0], instance.outline[1]]
        else:
            region[:] = [
                float(boxes[:, 0].min()),
                float(boxes[:, 1].min()),
                float((boxes[:, 0] + boxes[:, 2]).max()),
                float((boxes[:, 1] + boxes[:, 3]).max()),
            ]
        for index, box in enumerate(boxes):
            if not _intersects(
                box,
                np.asarray(
                    [region[0], region[1], region[2] - region[0], region[3] - region[1]]
                ),
                tolerance,
            ):
                continue
            if index not in selected:
                selected.add(index)
                region[0] = min(region[0], float(box[0]))
                region[1] = min(region[1], float(box[1]))
                region[2] = max(region[2], float(box[0] + box[2]))
                region[3] = max(region[3], float(box[1] + box[3]))

    return np.asarray(sorted(selected), dtype=np.int64), region


def _repair_once(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    seed: list[int],
    *,
    expansion: float,
    resolution: float,
    time_limit_ms: int,
    tolerance: float,
) -> np.ndarray | None:
    local_indices, region = _local_region(
        boxes,
        seed,
        instance,
        expansion=expansion,
        tolerance=tolerance,
    )
    origin = region[:2].copy()
    local_boxes = boxes[local_indices] - np.asarray(
        [origin[0], origin[1], 0.0, 0.0]
    )
    local_outline = (
        float(region[2] - region[0]),
        float(region[3] - region[1]),
    )
    anchored = instance.is_fixed | instance.is_preplaced
    immutable = anchored[local_indices]
    packed = pack_fixed_outline(
        local_boxes,
        local_outline,
        time_limit_ms=time_limit_ms,
        immutable=immutable,
        boundary_codes=instance.boundary_code[local_indices],
        coordinate_resolution=resolution,
    )
    if packed is None:
        return None
    candidate = boxes.copy()
    candidate[local_indices] = packed.xywh + np.asarray(
        [origin[0], origin[1], 0.0, 0.0]
    )
    return candidate


def repair_dbu_overlaps(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    *,
    resolution: float = 0.001,
    expansions: tuple[float, ...] = (0.25, 0.5, 1.0),
    max_move_fraction: float = 0.5,
    time_limit_ms: int = 2000,
) -> DbuPackingResult:
    """Resolve overlap without changing quantized shapes or anchored positions.

    The original instance is the sole source of area, fixed/preplaced and boundary
    semantics. Only a fully hard-feasible, non-worsening soft result is accepted.
    """

    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if max_move_fraction < 0.0:
        raise ValueError("max_move_fraction must be non-negative")
    if not expansions or any(value < 0.0 for value in expansions):
        raise ValueError("expansions must contain non-negative values")
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be positive")
    incoming = np.asarray(boxes, dtype=np.float64)
    if incoming.shape != (instance.block_count, 4):
        raise ValueError("boxes must have shape (blocks, 4)")
    if not np.isfinite(incoming).all():
        raise ValueError("boxes must be finite")

    tolerance = resolution * 0.5
    snapped = incoming.copy()
    movable = ~(instance.is_fixed | instance.is_preplaced)
    snapped[movable, :2] = (
        np.rint(snapped[movable, :2] / resolution) * resolution
    )
    (
        initial_report,
        initial_soft,
        _initial_grid,
    ) = _audit(snapped, instance, resolution)
    initial_overlap = len(initial_report.overlap_pairs)
    grid_failure = bool(
        np.any(
            np.abs(
                np.rint(snapped[:, 2:] / resolution) * resolution
                - snapped[:, 2:]
            )
            > 1.0e-6
        )
    )
    anchored_shape_failure = bool(
        np.any(
            np.abs(snapped[~movable, :2] - incoming[~movable, :2])
            > 1.0e-6
        )
    )
    if grid_failure or anchored_shape_failure:
        report, soft, residual = _audit(snapped, instance, resolution)
        return DbuPackingResult(
            boxes=incoming.copy(),
            accepted=False,
            route="grid_contract_failure",
            overlap_before=len(report.overlap_pairs),
            overlap_after=len(report.overlap_pairs),
            area_violations=len(report.area_violations),
            fixed_violations=len(report.fixed_violations),
            preplaced_violations=len(report.preplaced_violations),
            grouping_before=soft.grouping,
            grouping_after=soft.grouping,
            boundary_before=soft.boundary,
            boundary_after=soft.boundary,
            max_grid_residual=residual,
            max_displacement=0.0,
            moved_blocks=0,
            solver_calls=0,
        )

    if initial_report.is_feasible():
        return DbuPackingResult(
            boxes=snapped,
            accepted=True,
            route="already_feasible",
            overlap_before=0,
            overlap_after=0,
            area_violations=0,
            fixed_violations=0,
            preplaced_violations=0,
            grouping_before=initial_soft.grouping,
            grouping_after=initial_soft.grouping,
            boundary_before=initial_soft.boundary,
            boundary_after=initial_soft.boundary,
            max_grid_residual=0.0,
            max_displacement=float(
                np.max(np.abs(snapped[:, :2] - incoming[:, :2]), initial=0.0)
            ),
            moved_blocks=0,
            solver_calls=0,
        )

    if (
        initial_report.area_violations
        or initial_report.fixed_violations
        or initial_report.preplaced_violations
        or initial_report.aspect_violations
    ):
        return DbuPackingResult(
            boxes=incoming.copy(),
            accepted=False,
            route="non_geometric_constraint_failure",
            overlap_before=initial_overlap,
            overlap_after=initial_overlap,
            area_violations=len(initial_report.area_violations),
            fixed_violations=len(initial_report.fixed_violations),
            preplaced_violations=len(initial_report.preplaced_violations),
            grouping_before=initial_soft.grouping,
            grouping_after=initial_soft.grouping,
            boundary_before=initial_soft.boundary,
            boundary_after=initial_soft.boundary,
            max_grid_residual=0.0,
            max_displacement=0.0,
            moved_blocks=0,
            solver_calls=0,
        )

    current = snapped
    current_report = initial_report
    current_soft = initial_soft
    solver_calls = 0
    span = max(
        float((snapped[:, 0] + snapped[:, 2]).max() - snapped[:, 0].min()),
        float((snapped[:, 1] + snapped[:, 3]).max() - snapped[:, 1].min()),
        1.0,
    )
    for _ in range(instance.block_count):
        components = _overlap_components(
            instance.block_count,
            current_report.overlap_pairs,
        )
        if not components:
            break
        seed = components[0]
        improved = False
        for expansion in expansions:
            candidate = _repair_once(
                current,
                instance,
                seed,
                expansion=float(expansion),
                resolution=resolution,
                time_limit_ms=time_limit_ms,
                tolerance=tolerance,
            )
            solver_calls += 1
            if candidate is None:
                continue
            report, soft, _grid = _audit(candidate, instance, resolution)
            displacement = np.max(
                np.abs(candidate[:, :2] - snapped[:, :2]),
                initial=0.0,
            )
            if (
                len(report.overlap_pairs) >= len(current_report.overlap_pairs)
                or soft.grouping > initial_soft.grouping
                or soft.boundary > initial_soft.boundary
                or displacement > max_move_fraction * span
            ):
                continue
            current = candidate
            current_report = report
            current_soft = soft
            improved = True
            break
        if not improved:
            break

    accepted = bool(
        current_report.is_feasible()
        and current_soft.grouping <= initial_soft.grouping
        and current_soft.boundary <= initial_soft.boundary
    )
    result_boxes = current if accepted else incoming.copy()
    if accepted:
        route = "integer_dbu_local_milp"
    elif len(current_report.overlap_pairs) < initial_overlap:
        route = "rejected_incomplete_repair"
    else:
        route = "rejected_no_feasible_packing"
    final_report, final_soft, residual = _audit(
        result_boxes,
        instance,
        resolution,
    )
    displacement = np.max(
        np.abs(result_boxes[:, :2] - snapped[:, :2]),
        initial=0.0,
    )
    return DbuPackingResult(
        boxes=result_boxes,
        accepted=accepted,
        route=route,
        overlap_before=initial_overlap,
        overlap_after=len(final_report.overlap_pairs),
        area_violations=len(final_report.area_violations),
        fixed_violations=len(final_report.fixed_violations),
        preplaced_violations=len(final_report.preplaced_violations),
        grouping_before=initial_soft.grouping,
        grouping_after=final_soft.grouping,
        boundary_before=initial_soft.boundary,
        boundary_after=final_soft.boundary,
        max_grid_residual=residual,
        max_displacement=float(displacement),
        moved_blocks=int(
            np.count_nonzero(
                np.linalg.norm(result_boxes[:, :2] - snapped[:, :2], axis=1)
                > tolerance
            )
        ),
        solver_calls=solver_calls,
    )


__all__ = ["DbuPackingResult", "repair_dbu_overlaps"]
