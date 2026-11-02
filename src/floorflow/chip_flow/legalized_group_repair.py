"""Target-free local cluster repair for already legalized placements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trinity.floorplan.geometry import component_labels, connected_components
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.soft import check_soft
from trinity.floorplan.types import FloorplanInstance, Placement


@dataclass(frozen=True)
class GroupRepairResult:
    """Repaired boxes and target-free change diagnostics."""

    boxes: np.ndarray
    moves: int
    grouping_before: int
    grouping_after: int
    boundary_before: int
    boundary_after: int
    b2b_length_before: float
    b2b_length_after: float
    p2b_length_before: float
    p2b_length_after: float
    feasible: bool


def _net_lengths(
    boxes: np.ndarray,
    instance: FloorplanInstance,
) -> tuple[float, float]:
    centers = boxes[:, :2] + boxes[:, 2:] / 2.0
    b2b = np.asarray(instance.b2b).reshape(-1, 3)
    p2b = np.asarray(instance.p2b).reshape(-1, 3)
    b2b_length = 0.0
    if len(b2b):
        left = b2b[:, 0].astype(int)
        right = b2b[:, 1].astype(int)
        valid = (left >= 0) & (right >= 0) & (left < len(boxes)) & (right < len(boxes))
        distance = np.abs(centers[left[valid]] - centers[right[valid]]).sum(axis=1)
        b2b_length = float((distance * b2b[valid, 2]).sum())
    p2b_length = 0.0
    if len(p2b):
        pin = p2b[:, 0].astype(int)
        block = p2b[:, 1].astype(int)
        pins = np.asarray(instance.pins_pos)
        valid = (
            (pin >= 0)
            & (block >= 0)
            & (pin < len(pins))
            & (block < len(boxes))
        )
        distance = np.abs(centers[block[valid]] - pins[pin[valid]]).sum(axis=1)
        p2b_length = float((distance * p2b[valid, 2]).sum())
    return b2b_length, p2b_length


def _target_free_checks(
    boxes: np.ndarray,
    instance: FloorplanInstance,
) -> tuple[bool, int, int, float, float]:
    placement = Placement(
        xywh=np.asarray(boxes, dtype=np.float64).copy(),
        instance=instance,
    )
    feasible = check_feasibility(placement).is_feasible()
    soft = check_soft(placement)
    b2b_length, p2b_length = _net_lengths(placement.xywh, instance)
    return feasible, int(soft.grouping), int(soft.boundary), b2b_length, p2b_length


def _contact_shifts(
    boxes: np.ndarray,
    main: list[int],
    component: list[int],
    max_move: float,
) -> list[tuple[float, float]]:
    component_x0 = float(boxes[component, 0].min())
    component_y0 = float(boxes[component, 1].min())
    component_x1 = float((boxes[component, 0] + boxes[component, 2]).max())
    component_y1 = float((boxes[component, 1] + boxes[component, 3]).max())
    component_width = component_x1 - component_x0
    component_height = component_y1 - component_y0
    shifts: dict[tuple[float, float], tuple[float, float]] = {}
    for index in main:
        x, y, width, height = (float(value) for value in boxes[index])
        y_positions = (
            y,
            y + height - component_height,
            float(
                np.clip(
                    component_y0,
                    y,
                    y + height - component_height,
                )
            )
            if height >= component_height
            else y,
        )
        x_positions = (
            x,
            x + width - component_width,
            float(
                np.clip(
                    component_x0,
                    x,
                    x + width - component_width,
                )
            )
            if width >= component_width
            else x,
        )
        candidates = [
            *((x + width - component_x0, target_y - component_y0) for target_y in y_positions),
            *((x - component_width - component_x0, target_y - component_y0) for target_y in y_positions),
            *((target_x - component_x0, y + height - component_y0) for target_x in x_positions),
            *((target_x - component_x0, y - component_height - component_y0) for target_x in x_positions),
        ]
        for dx, dy in candidates:
            if abs(dx) + abs(dy) <= max_move:
                key = (round(float(dx), 6), round(float(dy), 6))
                shifts[key] = (float(dx), float(dy))
    return sorted(shifts.values(), key=lambda shift: abs(shift[0]) + abs(shift[1]))


def repair_legalized_groups(
    boxes: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    *,
    max_move_fraction: float = 0.25,
    max_net_increase: float = 0.0,
) -> GroupRepairResult:
    """Attach split cluster components without moving protected blocks.

    Only fixed, preplaced or boundary-coded components are protected. Every move
    must retain hard feasibility and must not increase the target-free boundary
    violation count. Candidate selection uses only geometry and constraints.
    """

    if max_move_fraction <= 0.0:
        raise ValueError("max_move_fraction must be positive")
    if max_net_increase < 0.0:
        raise ValueError("max_net_increase must be non-negative")
    current = np.asarray(boxes, dtype=np.float64).copy()
    constraints = np.asarray(constraints)
    if current.ndim != 2 or current.shape[1] != 4:
        raise ValueError("boxes must have shape (blocks, 4)")
    if constraints.shape != (len(current), 5):
        raise ValueError("constraints must have shape (blocks, 5)")

    protected = (
        (constraints[:, 0] != 0)
        | (constraints[:, 1] != 0)
        | (constraints[:, 4] != 0)
    )
    span = max(
        float((current[:, 0] + current[:, 2]).max() - current[:, 0].min()),
        float((current[:, 1] + current[:, 3]).max() - current[:, 1].min()),
        1.0,
    )
    (
        _,
        initial_grouping,
        initial_boundary,
        initial_b2b,
        initial_p2b,
    ) = _target_free_checks(current, instance)
    boundary_limit = initial_boundary
    b2b_limit = initial_b2b * (1.0 + float(max_net_increase))
    p2b_limit = initial_p2b * (1.0 + float(max_net_increase))
    moves = 0
    cluster_ids = np.unique(constraints[:, 3].astype(int))
    for cluster_id in cluster_ids[cluster_ids > 0]:
        members = np.nonzero(constraints[:, 3].astype(int) == cluster_id)[0]
        if len(members) < 2:
            continue
        for _ in range(len(members)):
            labels = component_labels(current, members)
            components: dict[int, list[int]] = {}
            for index in members:
                components.setdefault(labels[int(index)], []).append(int(index))
            if len(components) <= 1:
                break
            groups = list(components.values())
            main = max(
                groups,
                key=lambda group: (
                    sum(bool(protected[index]) for index in group),
                    sum(
                        bool((constraints[index, 0] != 0) or (constraints[index, 1] != 0))
                        for index in group
                    ),
                    len(group),
                ),
            )
            accepted = False
            for component in sorted(
                (group for group in groups if group is not main),
                key=len,
                reverse=True,
            ):
                if any(protected[index] for index in component):
                    continue
                for dx, dy in _contact_shifts(
                    current,
                    main,
                    component,
                    max_move_fraction * span,
                ):
                    candidate = current.copy()
                    candidate[component, 0] += dx
                    candidate[component, 1] += dy
                    feasible, _grouping, boundary, b2b_length, p2b_length = _target_free_checks(
                        candidate,
                        instance,
                    )
                    if (
                        feasible
                        and connected_components(candidate, members)
                        < len(components)
                        and boundary <= boundary_limit
                        and b2b_length <= b2b_limit
                        and p2b_length <= p2b_limit
                    ):
                        current = candidate
                        boundary_limit = boundary
                        moves += 1
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                break
    (
        feasible,
        grouping_after,
        boundary_after,
        b2b_after,
        p2b_after,
    ) = _target_free_checks(current, instance)
    return GroupRepairResult(
        boxes=current,
        moves=moves,
        grouping_before=initial_grouping,
        grouping_after=grouping_after,
        boundary_before=initial_boundary,
        boundary_after=boundary_after,
        b2b_length_before=initial_b2b,
        b2b_length_after=b2b_after,
        p2b_length_before=initial_p2b,
        p2b_length_after=p2b_after,
        feasible=feasible,
    )


__all__ = ["GroupRepairResult", "repair_legalized_groups"]
