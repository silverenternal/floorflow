"""Multi-block cluster reflow actions on a legalized placement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from trinity.floorplan.geometry import component_labels, connected_components
from trinity.floorplan.types import FloorplanInstance

from floorflow.chip_flow.legalized_group_actions import (
    _dense_contact_shifts,
    _edge_fractions,
    _group_bbox_area,
)
from floorflow.chip_flow.legalized_group_repair import _target_free_checks

CLUSTER_REFLOW_FEATURE_NAMES = (
    "log_moved_blocks",
    "log_main_blocks_before",
    "log_component_count",
    "log_unattached_blocks",
    "mean_move_l1_over_span",
    "max_move_l1_over_span",
    "horizontal_fraction",
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
    "protected_outside_main_fraction",
    "attached_fraction",
)


@dataclass(frozen=True)
class ClusterReflowAction:
    """One composite action moving several unprotected cluster blocks."""

    cluster_id: int
    moves: tuple[tuple[int, float, float], ...]
    main_anchor: int
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
            "moves": [
                [int(index), float(dx), float(dy)]
                for index, dx, dy in self.moves
            ],
            "main_anchor": int(self.main_anchor),
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


def _component_groups(
    boxes: np.ndarray,
    members: np.ndarray,
) -> dict[int, list[int]]:
    labels = component_labels(boxes, members)
    groups: dict[int, list[int]] = {}
    for index in members:
        groups.setdefault(labels[int(index)], []).append(int(index))
    return groups


def _main_group(
    groups: dict[int, list[int]],
    protected: np.ndarray,
    constraints: np.ndarray,
) -> tuple[list[int], int]:
    main = max(
        groups.values(),
        key=lambda group: (
            sum(bool(protected[index]) for index in group),
            sum(
                bool(
                    (constraints[index, 0] != 0)
                    or (constraints[index, 1] != 0)
                )
                for index in group
            ),
            len(group),
        ),
    )
    anchor = next((index for index in main if protected[index]), main[0])
    return main, int(anchor)


def _current_main(
    boxes: np.ndarray,
    members: np.ndarray,
    anchor: int,
) -> list[int]:
    labels = component_labels(boxes, members)
    anchor_label = labels[int(anchor)]
    return [
        int(index)
        for index in members
        if labels[int(index)] == anchor_label
    ]


def _block_distance(boxes: np.ndarray, index: int, main: list[int]) -> float:
    center = boxes[index, :2] + boxes[index, 2:] / 2.0
    main_centers = boxes[main, :2] + boxes[main, 2:] / 2.0
    return float(np.linalg.norm(main_centers - center, axis=1).min())


def _block_degree(instance: FloorplanInstance, index: int) -> float:
    b2b = np.asarray(instance.b2b).reshape(-1, 3)
    p2b = np.asarray(instance.p2b).reshape(-1, 3)
    degree = 0.0
    if len(b2b):
        degree += float(
            b2b[
                (b2b[:, 0].astype(int) == index)
                | (b2b[:, 1].astype(int) == index),
                2,
            ].sum()
        )
    if len(p2b):
        degree += float(p2b[p2b[:, 1].astype(int) == index, 2].sum())
    return degree


def _candidate_orders(
    boxes: np.ndarray,
    movable: list[int],
    main: list[int],
    instance: FloorplanInstance,
) -> list[list[int]]:
    distance = {
        index: _block_distance(boxes, index, main)
        for index in movable
    }
    degree = {
        index: _block_degree(instance, index)
        for index in movable
    }
    orders = [
        sorted(movable, key=lambda index: (degree[index], distance[index], index)),
        sorted(movable, key=lambda index: (distance[index], degree[index], index)),
        sorted(movable, key=lambda index: (-distance[index], index)),
    ]
    unique = []
    for order in orders:
        if order not in unique:
            unique.append(order)
    return unique


def _composite_features(
    *,
    before: np.ndarray,
    after: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    cluster_id: int,
    initial_main: list[int],
    final_main: list[int],
    initially_movable: list[int],
    moves: tuple[tuple[int, float, float], ...],
    grouping_before: int,
    grouping_after: int,
    boundary_before: int,
    boundary_after: int,
    b2b_before: float,
    b2b_after: float,
    p2b_before: float,
    p2b_after: float,
) -> np.ndarray:
    span = max(
        float((before[:, 0] + before[:, 2]).max() - before[:, 0].min()),
        float((before[:, 1] + before[:, 3]).max() - before[:, 1].min()),
        1.0,
    )
    members = np.nonzero(constraints[:, 3].astype(int) == cluster_id)[0]
    moved = {index for index, _dx, _dy in moves}
    displacement = np.asarray([abs(dx) + abs(dy) for _index, dx, dy in moves])
    horizontal = np.asarray(
        [abs(dx) >= abs(dy) for _index, dx, dy in moves],
        dtype=np.float64,
    )
    protected = (
        (constraints[:, 0] != 0)
        | (constraints[:, 1] != 0)
        | (constraints[:, 4] != 0)
    )
    protected_members = members[protected[members]]
    protected_outside = np.asarray(
        [index not in final_main for index in protected_members],
        dtype=np.float64,
    )
    bbox_before = _group_bbox_area(before, members) / span**2
    bbox_after = _group_bbox_area(after, members) / span**2
    b2b_cross, p2b_cross, b2b_main, p2b_main = _edge_fractions(
        instance,
        moved,
        set(final_main),
    )
    attached = len(set(final_main) - set(initial_main))
    candidates = max(len(initially_movable), 1)
    return np.asarray(
        [
            math.log1p(len(moves)),
            math.log1p(len(initial_main)),
            math.log1p(grouping_before),
            math.log1p(max(len(initially_movable) - attached, 0)),
            float(displacement.mean() / span),
            float(displacement.max() / span),
            float(horizontal.mean()),
            float(boundary_after - boundary_before),
            b2b_after / max(b2b_before, 1.0e-12) - 1.0,
            p2b_after / max(p2b_before, 1.0e-12) - 1.0,
            b2b_cross,
            p2b_cross,
            b2b_main,
            p2b_main,
            bbox_before,
            bbox_after,
            bbox_after / max(bbox_before, 1.0e-12) - 1.0,
            math.log1p(cluster_id),
            float(protected_outside.mean()) if len(protected_outside) else 0.0,
            attached / candidates,
        ],
        dtype=np.float64,
    )


def _reflow_order(
    before: np.ndarray,
    members: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    protected: np.ndarray,
    initial_main: list[int],
    anchor: int,
    initial_components: int,
    order: list[int],
    *,
    max_move_fraction: float,
    max_net_increase: float,
    boundary_limit: int,
    coordinate_resolution_um: float,
) -> tuple[np.ndarray, tuple[tuple[int, float, float], ...]] | None:
    current = before.copy()
    main = list(initial_main)
    span = max(
        float((before[:, 0] + before[:, 2]).max() - before[:, 0].min()),
        float((before[:, 1] + before[:, 3]).max() - before[:, 1].min()),
        1.0,
    )
    _, _, _, b2b_before, p2b_before = _target_free_checks(before, instance)
    b2b_limit = b2b_before * (1.0 + max_net_increase)
    p2b_limit = p2b_before * (1.0 + max_net_increase)
    moves: list[tuple[int, float, float]] = []
    for index in order:
        if index in main:
            continue
        accepted = False
        for dx, dy in _dense_contact_shifts(
            current,
            main,
            [index],
            max_move_fraction * span,
        ):
            candidate = current.copy()
            candidate[index, 0] = (
                np.round((candidate[index, 0] + dx) / coordinate_resolution_um)
                * coordinate_resolution_um
            )
            candidate[index, 1] = (
                np.round((candidate[index, 1] + dy) / coordinate_resolution_um)
                * coordinate_resolution_um
            )
            quantized_dx = float(candidate[index, 0] - current[index, 0])
            quantized_dy = float(candidate[index, 1] - current[index, 1])
            feasible, _, boundary, b2b_length, p2b_length = _target_free_checks(
                candidate,
                instance,
            )
            if (
                not feasible
                or boundary > boundary_limit
                or b2b_length > b2b_limit
                or p2b_length > p2b_limit
            ):
                continue
            candidate_components = connected_components(candidate, members)
            candidate_main = _current_main(candidate, members, anchor)
            if (
                candidate_components <= initial_components
                and index in candidate_main
            ):
                current = candidate
                main = candidate_main
                moves.append((int(index), quantized_dx, quantized_dy))
                accepted = True
                break
        if not accepted:
            continue
    if not moves or connected_components(current, members) >= initial_components:
        return None
    return current, tuple(moves)


def enumerate_cluster_reflow_actions(
    boxes: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    *,
    max_move_fraction: float = 0.25,
    max_net_increase: float = 0.02,
    max_actions_per_cluster: int = 16,
    coordinate_resolution_um: float = 0.001,
) -> list[ClusterReflowAction]:
    """Enumerate safe composite actions that reflow split cluster blocks."""

    if max_move_fraction <= 0.0:
        raise ValueError("max_move_fraction must be positive")
    if max_net_increase < 0.0:
        raise ValueError("max_net_increase must be non-negative")
    if max_actions_per_cluster <= 0:
        raise ValueError("max_actions_per_cluster must be positive")
    if coordinate_resolution_um <= 0.0:
        raise ValueError("coordinate resolution must be positive")
    before = np.asarray(boxes, dtype=np.float64).copy()
    constraints = np.asarray(constraints)
    if before.ndim != 2 or before.shape[1] != 4:
        raise ValueError("boxes must have shape (blocks, 4)")
    if constraints.shape != (len(before), 5):
        raise ValueError("constraints must have shape (blocks, 5)")
    feasible, _, boundary, b2b_length, p2b_length = _target_free_checks(
        before,
        instance,
    )
    if not feasible:
        raise ValueError("cluster reflow requires a hard-feasible placement")

    protected = (
        (constraints[:, 0] != 0)
        | (constraints[:, 1] != 0)
        | (constraints[:, 4] != 0)
    )
    actions: list[ClusterReflowAction] = []
    seen_states: set[tuple[float, ...]] = set()
    cluster_ids = np.unique(constraints[:, 3].astype(int))
    for cluster_id in cluster_ids[cluster_ids > 0]:
        members = np.nonzero(constraints[:, 3].astype(int) == cluster_id)[0]
        if len(members) < 2:
            continue
        groups = _component_groups(before, members)
        if len(groups) <= 1:
            continue
        initial_main, anchor = _main_group(groups, protected, constraints)
        initially_movable = [
            int(index)
            for index in members
            if index not in initial_main and not protected[index]
        ]
        if not initially_movable:
            continue
        orders = _candidate_orders(
            before,
            initially_movable,
            initial_main,
            instance,
        )
        cluster_actions = 0
        for order in orders:
            reflowed = _reflow_order(
                before,
                members,
                constraints,
                instance,
                protected,
                initial_main,
                anchor,
                len(groups),
                order,
                max_move_fraction=max_move_fraction,
                max_net_increase=max_net_increase,
                boundary_limit=boundary,
        coordinate_resolution_um=coordinate_resolution_um,
            )
            if reflowed is None:
                continue
            after, moves = reflowed
            key = tuple(np.round(after.ravel(), 6).tolist())
            if key in seen_states:
                continue
            _feasible_after, grouping_after, boundary_after, b2b_after, p2b_after = (
                _target_free_checks(after, instance)
            )
            final_main = _current_main(after, members, anchor)
            features = _composite_features(
                before=before,
                after=after,
                constraints=constraints,
                instance=instance,
                cluster_id=int(cluster_id),
                initial_main=initial_main,
                final_main=final_main,
                initially_movable=initially_movable,
                moves=moves,
                grouping_before=len(groups),
                grouping_after=grouping_after,
                boundary_before=boundary,
                boundary_after=boundary_after,
                b2b_before=b2b_length,
                b2b_after=b2b_after,
                p2b_before=p2b_length,
                p2b_after=p2b_after,
            )
            actions.append(
                ClusterReflowAction(
                    cluster_id=int(cluster_id),
                    moves=moves,
                    main_anchor=anchor,
                    grouping_before=len(groups),
                    grouping_after=grouping_after,
                    boundary_before=boundary,
                    boundary_after=boundary_after,
                    b2b_length_before=b2b_length,
                    b2b_length_after=b2b_after,
                    p2b_length_before=p2b_length,
                    p2b_length_after=p2b_after,
                    features=features,
                    boxes=after,
                )
            )
            seen_states.add(key)
            cluster_actions += 1
            if cluster_actions >= max_actions_per_cluster:
                break
    return sorted(
        actions,
        key=lambda action: (
            sum(abs(dx) + abs(dy) for _index, dx, dy in action.moves),
            action.cluster_id,
            action.moves,
        ),
    )


def apply_cluster_reflow_action(
    boxes: np.ndarray,
    action: ClusterReflowAction,
) -> np.ndarray:
    """Apply one composite action to the original layout."""

    result = np.asarray(boxes, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("boxes must have shape (blocks, 4)")
    for index, dx, dy in action.moves:
        if index < 0 or index >= len(result):
            raise ValueError("reflow move index is out of range")
        result[index, 0] += dx
        result[index, 1] += dy
    return result


def cluster_reflow_action_key(features: np.ndarray) -> tuple[float, float, float]:
    """Return the frozen target-free ranking key for a reflow action."""

    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (len(CLUSTER_REFLOW_FEATURE_NAMES),):
        raise ValueError("cluster reflow feature width mismatch")
    indices = [
        CLUSTER_REFLOW_FEATURE_NAMES.index(name)
        for name in (
            "b2b_relative_delta",
            "p2b_relative_delta",
            "mean_move_l1_over_span",
        )
    ]
    return tuple(float(vector[index]) for index in indices)


def select_cluster_reflow_action(
    actions: list[ClusterReflowAction],
) -> ClusterReflowAction | None:
    """Select the frozen safe action, preferring net preservation."""

    if not actions:
        return None
    return min(actions, key=lambda action: cluster_reflow_action_key(action.features))


__all__ = [
    "CLUSTER_REFLOW_FEATURE_NAMES",
    "ClusterReflowAction",
    "apply_cluster_reflow_action",
    "cluster_reflow_action_key",
    "enumerate_cluster_reflow_actions",
    "select_cluster_reflow_action",
]
