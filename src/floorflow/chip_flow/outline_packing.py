"""Deterministic fixed-outline packing for already-shaped chip blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp

_EPS = 1.0e-7


@dataclass(frozen=True)
class OutlinePackingResult:
    xywh: np.ndarray
    solver: str
    max_displacement: float


def _solver(time_limit_ms: int):
    # SCIP is materially more stable for the integer relation/disjunction
    # models used by chip packing. CBC has crashed on the larger local models.
    for name in ("SCIP", "CBC"):
        solver = pywraplp.Solver.CreateSolver(name)
        if solver is not None:
            solver.SetTimeLimit(max(1, int(time_limit_ms)))
            return solver, name
    return None, ""


def _valid_packing(xywh: np.ndarray, outline: tuple[float, float]) -> bool:
    if not np.isfinite(xywh).all():
        return False
    width, height = float(outline[0]), float(outline[1])
    if (
        (xywh[:, 0] < -_EPS).any()
        or (xywh[:, 1] < -_EPS).any()
        or (xywh[:, 0] + xywh[:, 2] > width + _EPS).any()
        or (xywh[:, 1] + xywh[:, 3] > height + _EPS).any()
    ):
        return False
    for i in range(len(xywh) - 1):
        for j in range(i + 1, len(xywh)):
            overlap_x = min(
                float(xywh[i, 0] + xywh[i, 2]),
                float(xywh[j, 0] + xywh[j, 2]),
            ) - max(float(xywh[i, 0]), float(xywh[j, 0]))
            overlap_y = min(
                float(xywh[i, 1] + xywh[i, 3]),
                float(xywh[j, 1] + xywh[j, 3]),
            ) - max(float(xywh[i, 1]), float(xywh[j, 1]))
            if overlap_x > _EPS and overlap_y > _EPS:
                return False
    return True


def pack_fixed_outline(
    boxes: np.ndarray,
    outline: tuple[float, float],
    *,
    time_limit_ms: int = 2000,
    rotatable: np.ndarray | None = None,
    immutable: np.ndarray | None = None,
    boundary_codes: np.ndarray | None = None,
    coordinate_resolution: float | None = None,
) -> OutlinePackingResult | None:
    """Pack fixed ``xywh`` rectangles inside ``outline``.

    The MILP uses four binary disjunctions per pair. It is intended as a small,
    explicit outline-feasibility fallback; scale-pack remains the primary
    legalizer. No shape, area or orientation is changed.
    """

    incoming = np.asarray(boxes, dtype=np.float64)
    if incoming.ndim != 2 or incoming.shape[1] != 4 or not len(incoming):
        raise ValueError("boxes must have shape (N, 4)")
    if not np.isfinite(incoming).all() or (incoming[:, 2:] <= 0.0).any():
        raise ValueError("box dimensions must be finite and positive")
    width, height = float(outline[0]), float(outline[1])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("outline dimensions must be positive")
    oversized_without_rotation = (
        (incoming[:, 2] > width + _EPS).any()
        or (incoming[:, 3] > height + _EPS).any()
    ) and (rotatable is None or not rotatable.any())
    if oversized_without_rotation:
        return None

    if coordinate_resolution is not None:
        return _pack_integer_grid(
            incoming,
            (width, height),
            time_limit_ms=time_limit_ms,
            immutable=immutable,
            boundary_codes=boundary_codes,
            coordinate_resolution=coordinate_resolution,
        )

    solver, solver_name = _solver(time_limit_ms)
    if solver is None:
        return None
    n = len(incoming)
    if rotatable is None:
        rotatable = np.zeros(n, dtype=bool)
    rotatable = np.asarray(rotatable, dtype=bool)
    if rotatable.shape != (n,):
        raise ValueError("rotatable must have shape (N,)")
    span_x = width + float(incoming[:, 2].sum())
    span_y = height + float(incoming[:, 3].sum())
    displacement_bound_x = span_x + float(np.abs(incoming[:, 0]).max())
    displacement_bound_y = span_y + float(np.abs(incoming[:, 1]).max())
    x = [
        solver.NumVar(
            0.0,
            max(
                0.0,
                width
                - min(float(incoming[i, 2]), float(incoming[i, 3]))
                if rotatable[i]
                else width - float(incoming[i, 2]),
            ),
            "",
        )
        for i in range(n)
    ]
    y = [
        solver.NumVar(
            0.0,
            max(
                0.0,
                height
                - min(float(incoming[i, 2]), float(incoming[i, 3]))
                if rotatable[i]
                else height - float(incoming[i, 3]),
            ),
            "",
        )
        for i in range(n)
    ]
    rotate = [
        solver.BoolVar("") if rotatable[i] else None
        for i in range(n)
    ]
    width_expr = [
        float(incoming[i, 2])
        + (
            (float(incoming[i, 3]) - float(incoming[i, 2])) * rotate[i]
            if rotate[i] is not None
            else 0.0
        )
        for i in range(n)
    ]
    height_expr = [
        float(incoming[i, 3])
        + (
            (float(incoming[i, 2]) - float(incoming[i, 3])) * rotate[i]
            if rotate[i] is not None
            else 0.0
        )
        for i in range(n)
    ]
    dx = [
        solver.NumVar(0.0, displacement_bound_x, "")
        for _ in range(n)
    ]
    dy = [
        solver.NumVar(0.0, displacement_bound_y, "")
        for _ in range(n)
    ]
    objective = solver.Objective()
    objective.SetMinimization()
    for i in range(n):
        objective.SetCoefficient(dx[i], 1.0 / max(width, 1.0e-9))
        objective.SetCoefficient(dy[i], 1.0 / max(height, 1.0e-9))
        # |x - x0| <= dx and |y - y0| <= dy.
        solver.Add(dx[i] >= x[i] - float(incoming[i, 0]))
        solver.Add(dx[i] >= float(incoming[i, 0]) - x[i])
        solver.Add(dy[i] >= y[i] - float(incoming[i, 1]))
        solver.Add(dy[i] >= float(incoming[i, 1]) - y[i])
        solver.Add(x[i] + width_expr[i] <= width)
        solver.Add(y[i] + height_expr[i] <= height)

    big_x = span_x + float(incoming[:, 2].max())
    big_y = span_y + float(incoming[:, 3].max())
    for i in range(n - 1):
        for j in range(i + 1, n):
            bx_left = solver.BoolVar("")
            bx_right = solver.BoolVar("")
            by_below = solver.BoolVar("")
            by_above = solver.BoolVar("")
            solver.Add(
                x[i] + width_expr[i]
                <= x[j] + big_x * (1.0 - bx_left)
            )
            solver.Add(
                x[j] + width_expr[j]
                <= x[i] + big_x * (1.0 - bx_right)
            )
            solver.Add(
                y[i] + height_expr[i]
                <= y[j] + big_y * (1.0 - by_below)
            )
            solver.Add(
                y[j] + height_expr[j]
                <= y[i] + big_y * (1.0 - by_above)
            )
            solver.Add(bx_left + bx_right + by_below + by_above >= 1)

    status = solver.Solve()
    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ):
        return None
    packed = incoming.copy()
    packed[:, 0] = [variable.solution_value() for variable in x]
    packed[:, 1] = [variable.solution_value() for variable in y]
    for i, variable in enumerate(rotate):
        if variable is not None and variable.solution_value() > 0.5:
            packed[i, 2], packed[i, 3] = packed[i, 3], packed[i, 2]
    if not _valid_packing(packed, outline):
        return None
    displacement = np.max(
        np.abs(packed[:, :2] - incoming[:, :2]),
        initial=0.0,
    )
    return OutlinePackingResult(
        xywh=packed,
        solver=solver_name,
        max_displacement=float(displacement),
    )


def _pack_integer_grid(
    incoming: np.ndarray,
    outline: tuple[float, float],
    *,
    time_limit_ms: int,
    immutable: np.ndarray | None,
    boundary_codes: np.ndarray | None,
    coordinate_resolution: float,
) -> OutlinePackingResult | None:
    """MILP companion for fixed shapes on a declared integer DBU grid."""

    if coordinate_resolution <= 0.0:
        raise ValueError("coordinate_resolution must be positive")
    count = len(incoming)
    if immutable is None:
        immutable = np.zeros(count, dtype=bool)
    immutable = np.asarray(immutable, dtype=bool)
    if immutable.shape != (count,):
        raise ValueError("immutable must have shape (N,)")
    if boundary_codes is None:
        boundary_codes = np.zeros(count, dtype=np.int64)
    boundary_codes = np.asarray(boundary_codes, dtype=np.int64)
    if boundary_codes.shape != (count,):
        raise ValueError("boundary_codes must have shape (N,)")

    resolution = float(coordinate_resolution)
    size_units = np.rint(incoming[:, 2:] / resolution).astype(np.int64)
    size_residual = np.abs(
        size_units.astype(np.float64) * resolution - incoming[:, 2:]
    )
    if np.any(size_residual > 1.0e-6):
        return None
    position_units = np.rint(incoming[:, :2] / resolution).astype(np.int64)
    position_residual = np.abs(
        position_units.astype(np.float64) * resolution - incoming[:, :2]
    )
    if np.any(position_residual[immutable] > 1.0e-6):
        return None
    width_units = round(outline[0] / resolution)
    height_units = round(outline[1] / resolution)
    if np.any(size_units[:, 0] > width_units):
        return None
    if np.any(size_units[:, 1] > height_units):
        return None

    solver, solver_name = _solver(time_limit_ms)
    if solver is None:
        return None
    x: list = []
    y: list = []
    for index in range(count):
        x_hi = width_units - int(size_units[index, 0])
        y_hi = height_units - int(size_units[index, 1])
        if immutable[index]:
            if (
                position_units[index, 0] < 0
                or position_units[index, 1] < 0
                or position_units[index, 0] > x_hi
                or position_units[index, 1] > y_hi
            ):
                return None
            x.append(
                solver.IntVar(
                    int(position_units[index, 0]),
                    int(position_units[index, 0]),
                    "",
                )
            )
            y.append(
                solver.IntVar(
                    int(position_units[index, 1]),
                    int(position_units[index, 1]),
                    "",
                )
            )
        else:
            x.append(solver.IntVar(0, x_hi, ""))
            y.append(solver.IntVar(0, y_hi, ""))

    dx = [solver.IntVar(0, width_units, "") for _ in range(count)]
    dy = [solver.IntVar(0, height_units, "") for _ in range(count)]
    objective = solver.Objective()
    objective.SetMinimization()
    for index in range(count):
        objective.SetCoefficient(dx[index], 1.0 / max(width_units, 1))
        objective.SetCoefficient(dy[index], 1.0 / max(height_units, 1))
        solver.Add(dx[index] >= x[index] - int(position_units[index, 0]))
        solver.Add(dx[index] >= int(position_units[index, 0]) - x[index])
        solver.Add(dy[index] >= y[index] - int(position_units[index, 1]))
        solver.Add(dy[index] >= int(position_units[index, 1]) - y[index])
        code = int(boundary_codes[index])
        if code & 1:
            solver.Add(x[index] == 0)
        if code & 2:
            solver.Add(x[index] == x_hi)
        if code & 8:
            solver.Add(y[index] == 0)
        if code & 4:
            solver.Add(y[index] == y_hi)

    for left in range(count - 1):
        for right in range(left + 1, count):
            if immutable[left] and immutable[right]:
                if _unit_overlap(
                    position_units[left],
                    size_units[left],
                    position_units[right],
                    size_units[right],
                ):
                    return None
                continue
            before_x = solver.BoolVar("")
            after_x = solver.BoolVar("")
            before_y = solver.BoolVar("")
            after_y = solver.BoolVar("")
            solver.Add(
                x[left] + int(size_units[left, 0])
                <= x[right] + width_units * (1 - before_x)
            )
            solver.Add(
                x[right] + int(size_units[right, 0])
                <= x[left] + width_units * (1 - after_x)
            )
            solver.Add(
                y[left] + int(size_units[left, 1])
                <= y[right] + height_units * (1 - before_y)
            )
            solver.Add(
                y[right] + int(size_units[right, 1])
                <= y[left] + height_units * (1 - after_y)
            )
            solver.Add(before_x + after_x + before_y + after_y >= 1)

    status = solver.Solve()
    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ):
        return None
    packed = incoming.copy()
    packed[:, 0] = [
        round(variable.solution_value()) * resolution
        for variable in x
    ]
    packed[:, 1] = [
        round(variable.solution_value()) * resolution
        for variable in y
    ]
    if not _valid_packing(packed, outline):
        return None
    displacement = np.max(
        np.abs(packed[:, :2] - incoming[:, :2]),
        initial=0.0,
    )
    return OutlinePackingResult(
        xywh=packed,
        solver=solver_name,
        max_displacement=float(displacement),
    )


def _unit_overlap(
    left_position: np.ndarray,
    left_size: np.ndarray,
    right_position: np.ndarray,
    right_size: np.ndarray,
) -> bool:
    """Whether two integer rectangles strictly overlap."""

    overlap_x = min(
        int(left_position[0] + left_size[0]),
        int(right_position[0] + right_size[0]),
    ) - max(int(left_position[0]), int(right_position[0]))
    overlap_y = min(
        int(left_position[1] + left_size[1]),
        int(right_position[1] + right_size[1]),
    ) - max(int(left_position[1]), int(right_position[1]))
    return overlap_x > 0 and overlap_y > 0


__all__ = ["OutlinePackingResult", "pack_fixed_outline"]
