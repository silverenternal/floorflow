"""Enumerate target-free local actions on an already legalized floorplan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trinity.floorplan.geometry import component_labels, connected_components
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.legalized_group_repair import _target_free_checks

FEATURE_NAMES = (
    "log_moved_blocks",
    "log_main_blocks",
    "log_component_count",
    "move_l1_over_span",
    "move_x_over_span",
    "move_y_over_span",
    "horizontal_axis",
    "boundary_delta",
    "b2b_relative_delta",
    "p2b_relative_delta",
    "moved_cross_b2b_fraction",
    "moved_cross_p2b_fraction",
    "main_b2b_fraction",
    "main_p2b_fraction",
    "group_bbox_area_before",
    "group_bbox_area_after",
    "group_bbox_area_relative_delta",
    "cluster_id_log1p",
)


@dataclass(frozen=True)
class LegalizedGroupAction:
    """A feasible rigid translation of one cluster component."""

    cluster_id: int
    moved_indices: tuple[int, ...]
    main_indices: tuple[int, ...]
    shift: tuple[float, float]
    grouping_before: int
    grouping_after: int
    boundary_before: int
    boundary_after: int
    b2b_length_before: float
    b2b_length_after: float
    p2b_length_before: float
    p2b_length_after: float
    features: np.ndarray
    boxes: np.ndarray

    def as_dict(self, *, include_boxes: bool = False) -> dict:
        payload = {
            "cluster_id": int(self.cluster_id),
            "moved_indices": list(self.moved_indices),
            "main_indices": list(self.main_indices),
            "shift": list(self.shift),
            "grouping_before": int(self.grouping_before),
            "grouping_after": int(self.grouping_after),
            "boundary_before": int(self.boundary_before),
            "boundary_after": int(self.boundary_after),
            "b2b_length_before": float(self.b2b_length_before),
            "b2b_length_after": float(self.b2b_length_after),
            "p2b_length_before": float(self.p2b_length_before),
            "p2b_length_after": float(self.p2b_length_after),
            "features": self.features.astype(float).tolist(),
        }
        if include_boxes:
            payload["boxes"] = self.boxes.astype(float).tolist()
        return payload


def _group_bbox_area(boxes: np.ndarray, indices: np.ndarray) -> float:
    x0 = float(boxes[indices, 0].min())
    y0 = float(boxes[indices, 1].min())
    x1 = float((boxes[indices, 0] + boxes[indices, 2]).max())
    y1 = float((boxes[indices, 1] + boxes[indices, 3]).max())
    return max((x1 - x0) * (y1 - y0), 0.0)


def _edge_fractions(
    instance: FloorplanInstance,
    moved: set[int],
    main: set[int],
) -> tuple[float, float, float, float]:
    b2b = np.asarray(instance.b2b).reshape(-1, 3)
    b2b_total = float(b2b[:, 2].sum()) if len(b2b) else 0.0
    b2b_cross = 0.0
    b2b_main = 0.0
    for left, right, weight in b2b:
        left_id, right_id = int(left), int(right)
        if left_id < 0 or right_id < 0:
            continue
        in_left = left_id in moved
        in_right = right_id in moved
        if in_left != in_right:
            b2b_cross += float(weight)
        if left_id in main or right_id in main:
            b2b_main += float(weight)

    p2b = np.asarray(instance.p2b).reshape(-1, 3)
    p2b_total = float(p2b[:, 2].sum()) if len(p2b) else 0.0
    p2b_cross = float(
        p2b[np.isin(p2b[:, 1].astype(int), list(moved)), 2].sum()
    ) if len(p2b) else 0.0
    p2b_main = float(
        p2b[np.isin(p2b[:, 1].astype(int), list(main)), 2].sum()
    ) if len(p2b) else 0.0
    b2b_denominator = max(b2b_total, 1.0e-12)
    p2b_denominator = max(p2b_total, 1.0e-12)
    return (
        b2b_cross / b2b_denominator,
        p2b_cross / p2b_denominator,
        b2b_main / b2b_denominator,
        p2b_main / p2b_denominator,
    )


def _overlap_positions(
    main_start: float,
    main_length: float,
    component_length: float,
    current_start: float,
) -> tuple[float, ...]:
    """Positions whose intervals overlap the main block over positive length."""

    if main_length <= 0.0 or component_length <= 0.0:
        return ()
    lower = main_start - component_length
    upper = main_start + main_length
    slack = min(main_length, component_length) * 1.0e-3
    positions = (
        main_start,
        main_start + main_length - component_length,
        main_start + 0.5 * (main_length - component_length),
        lower + slack,
        upper - slack,
        float(np.clip(current_start, lower + slack, upper - slack)),
    )
    return tuple(
        dict.fromkeys(
            float(position)
            for position in positions
            if lower + slack <= position <= upper - slack
        )
    )


def _dense_contact_shifts(
    boxes: np.ndarray,
    main: list[int],
    component: list[int],
    max_move: float,
) -> list[tuple[float, float]]:
    """Enumerate translations that put a component on any main-block edge."""

    component_x0 = float(boxes[component, 0].min())
    component_y0 = float(boxes[component, 1].min())
    component_x1 = float((boxes[component, 0] + boxes[component, 2]).max())
    component_y1 = float((boxes[component, 1] + boxes[component, 3]).max())
    component_width = component_x1 - component_x0
    component_height = component_y1 - component_y0
    shifts: dict[tuple[float, float], tuple[float, float]] = {}
    for index in main:
        x, y, width, height = (float(value) for value in boxes[index])
        y_positions = _overlap_positions(y, height, component_height, component_y0)
        x_positions = _overlap_positions(x, width, component_width, component_x0)
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


def _action_features(
    *,
    before: np.ndarray,
    after: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    cluster_id: int,
    moved: tuple[int, ...],
    main: tuple[int, ...],
    shift: tuple[float, float],
    grouping_before: int,
    grouping_after: int,
    boundary_delta: int,
    b2b_before: float,
    b2b_after: float,
    p2b_before: float,
    p2b_after: float,
) -> np.ndarray:
    span = max(
        float(
            max(
                (before[:, 0] + before[:, 2]).max() - before[:, 0].min(),
                (before[:, 1] + before[:, 3]).max() - before[:, 1].min(),
            )
        ),
        1.0,
    )
    members = np.nonzero(constraints[:, 3].astype(int) == cluster_id)[0]
    bbox_before = _group_bbox_area(before, members) / span**2
    bbox_after = _group_bbox_area(after, members) / span**2
    b2b_cross, p2b_cross, b2b_main, p2b_main = _edge_fractions(
        instance,
        set(moved),
        set(main),
    )
    dx, dy = shift
    return np.asarray(
        [
            np.log1p(len(moved)),
            np.log1p(len(main)),
            np.log1p(grouping_before),
            (abs(dx) + abs(dy)) / span,
            abs(dx) / span,
            abs(dy) / span,
            float(abs(dx) >= abs(dy)),
            float(boundary_delta),
            b2b_after / max(b2b_before, 1.0e-12) - 1.0,
            p2b_after / max(p2b_before, 1.0e-12) - 1.0,
            b2b_cross,
            p2b_cross,
            b2b_main,
            p2b_main,
            bbox_before,
            bbox_after,
            bbox_after / max(bbox_before, 1.0e-12) - 1.0,
            np.log1p(cluster_id),
        ],
        dtype=np.float64,
    )


def enumerate_legalized_group_actions(
    boxes: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    *,
    max_move_fraction: float = 0.25,
    max_net_increase: float = 0.02,
    max_actions_per_cluster: int = 64,
) -> list[LegalizedGroupAction]:
    """Enumerate safe one-step component moves from a legalized placement."""

    if max_move_fraction <= 0.0:
        raise ValueError("max_move_fraction must be positive")
    if max_net_increase < 0.0:
        raise ValueError("max_net_increase must be non-negative")
    if max_actions_per_cluster <= 0:
        raise ValueError("max_actions_per_cluster must be positive")
    before = np.asarray(boxes, dtype=np.float64).copy()
    constraints = np.asarray(constraints)
    if before.ndim != 2 or before.shape[1] != 4:
        raise ValueError("boxes must have shape (blocks, 4)")
    if constraints.shape != (len(before), 5):
        raise ValueError("constraints must have shape (blocks, 5)")

    protected = (
        (constraints[:, 0] != 0)
        | (constraints[:, 1] != 0)
        | (constraints[:, 4] != 0)
    )
    span = max(
        float((before[:, 0] + before[:, 2]).max() - before[:, 0].min()),
        float((before[:, 1] + before[:, 3]).max() - before[:, 1].min()),
        1.0,
    )
    feasible, _, boundary, b2b_length, p2b_length = _target_free_checks(
        before,
        instance,
    )
    if not feasible:
        raise ValueError("action enumeration requires a hard-feasible placement")
    b2b_limit = b2b_length * (1.0 + float(max_net_increase))
    p2b_limit = p2b_length * (1.0 + float(max_net_increase))
    actions: list[LegalizedGroupAction] = []
    seen: set[tuple[float, ...]] = set()

    cluster_ids = np.unique(constraints[:, 3].astype(int))
    for cluster_id in cluster_ids[cluster_ids > 0]:
        members = np.nonzero(constraints[:, 3].astype(int) == cluster_id)[0]
        if len(members) < 2:
            continue
        labels = component_labels(before, members)
        components: dict[int, list[int]] = {}
        for index in members:
            components.setdefault(labels[int(index)], []).append(int(index))
        if len(components) <= 1:
            continue
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
        cluster_actions = 0
        for component in sorted(
            (group for group in groups if group is not main),
            key=len,
            reverse=True,
        ):
            if any(protected[index] for index in component):
                continue
            for shift in _dense_contact_shifts(
                before,
                main,
                component,
                max_move_fraction * span,
            ):
                candidate = before.copy()
                candidate[component, 0] += shift[0]
                candidate[component, 1] += shift[1]
                key = tuple(np.round(candidate.ravel(), 6).tolist())
                if key in seen:
                    continue
                (
                    candidate_feasible,
                    _,
                    candidate_boundary,
                    candidate_b2b,
                    candidate_p2b,
                ) = _target_free_checks(candidate, instance)
                candidate_grouping = connected_components(candidate, members)
                if (
                    candidate_feasible
                    and candidate_grouping < len(components)
                    and candidate_boundary <= boundary
                    and candidate_b2b <= b2b_limit
                    and candidate_p2b <= p2b_limit
                ):
                    moved = tuple(component)
                    main_indices = tuple(main)
                    features = _action_features(
                        before=before,
                        after=candidate,
                        constraints=constraints,
                        instance=instance,
                        cluster_id=int(cluster_id),
                        moved=moved,
                        main=main_indices,
                        shift=shift,
                        grouping_before=len(components),
                        grouping_after=candidate_grouping,
                        boundary_delta=candidate_boundary - boundary,
                        b2b_before=b2b_length,
                        b2b_after=candidate_b2b,
                        p2b_before=p2b_length,
                        p2b_after=candidate_p2b,
                    )
                    actions.append(
                        LegalizedGroupAction(
                            cluster_id=int(cluster_id),
                            moved_indices=moved,
                            main_indices=main_indices,
                            shift=(float(shift[0]), float(shift[1])),
                            grouping_before=len(components),
                            grouping_after=candidate_grouping,
                            boundary_before=boundary,
                            boundary_after=candidate_boundary,
                            b2b_length_before=b2b_length,
                            b2b_length_after=candidate_b2b,
                            p2b_length_before=p2b_length,
                            p2b_length_after=candidate_p2b,
                            features=features,
                            boxes=candidate,
                        )
                    )
                    seen.add(key)
                    cluster_actions += 1
                    if cluster_actions >= max_actions_per_cluster:
                        break
            if cluster_actions >= max_actions_per_cluster:
                break
    return sorted(
        actions,
        key=lambda action: (
            abs(action.shift[0]) + abs(action.shift[1]),
            action.cluster_id,
            action.moved_indices,
        ),
    )


def apply_legalized_group_action(
    boxes: np.ndarray,
    action: LegalizedGroupAction,
) -> np.ndarray:
    """Apply an action to boxes without trusting its cached result tensor."""

    result = np.asarray(boxes, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("boxes must have shape (blocks, 4)")
    moved = np.asarray(action.moved_indices, dtype=np.int64)
    if len(moved) == 0 or moved.min() < 0 or moved.max() >= len(result):
        raise ValueError("action moved indices are out of range")
    result[moved, 0] += action.shift[0]
    result[moved, 1] += action.shift[1]
    return result


__all__ = [
    "FEATURE_NAMES",
    "LegalizedGroupAction",
    "apply_legalized_group_action",
    "enumerate_legalized_group_actions",
]
