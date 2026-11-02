"""Integer-DBU projection of a quantized, continuously legal chip layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp
from trinity.floorplan.geometry import component_labels
from trinity.floorplan.legalize.pack_relations import pair_relation
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.soft import check_soft
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.legalization import scale_pack_legalize_reproducible


@dataclass(frozen=True)
class DbuProjectionResult:
    boxes: np.ndarray
    accepted: bool
    route: str
    solver: str
    continuous_boxes: np.ndarray | None
    overlap_before: int
    overlap_after: int
    area_violations: int
    fixed_violations: int
    preplaced_violations: int
    aspect_violations: int
    grouping_before: int
    grouping_after: int
    boundary_before: int
    boundary_after: int
    max_grid_residual: float
    max_displacement: float
    moved_blocks: int
    contact_edges: int
    solver_seconds: float


def _solver(time_limit_ms: int):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        return None
    solver.SetTimeLimit(max(1, int(time_limit_ms)))
    return solver


def _quantized_instance(
    boxes: np.ndarray,
    instance: FloorplanInstance,
) -> FloorplanInstance:
    targets = np.asarray(instance.target_positions, dtype=np.float64).copy()
    fixed = instance.is_fixed
    preplaced = instance.is_preplaced
    targets[fixed, 2:] = boxes[fixed, 2:]
    targets[preplaced] = boxes[preplaced]
    areas = np.asarray(instance.area_targets, dtype=np.float64).copy()
    soft = instance.is_soft
    areas[soft] = boxes[soft, 2] * boxes[soft, 3]
    return FloorplanInstance(
        block_count=instance.block_count,
        area_targets=areas,
        constraints=instance.constraints,
        b2b=instance.b2b,
        p2b=instance.p2b,
        pins_pos=instance.pins_pos,
        target_positions=targets,
        gt_positions=instance.gt_positions,
        outline=instance.outline,
        aspect_bounds=instance.aspect_bounds,
    )


def _pair_contact_option(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    left: int,
    right: int,
) -> tuple[float, int, int, int] | None:
    y_overlap = min(
        boxes[left, 1] + boxes[left, 3],
        boxes[right, 1] + boxes[right, 3],
    ) - max(boxes[left, 1], boxes[right, 1])
    x_overlap = min(
        boxes[left, 0] + boxes[left, 2],
        boxes[right, 0] + boxes[right, 2],
    ) - max(boxes[left, 0], boxes[right, 0])
    candidates = []
    if y_overlap > 1.0e-7:
        candidates.extend(
            [
                (
                    abs(boxes[left, 0] + boxes[left, 2] - boxes[right, 0]),
                    0,
                    left,
                    right,
                ),
                (
                    abs(boxes[right, 0] + boxes[right, 2] - boxes[left, 0]),
                    0,
                    right,
                    left,
                ),
            ]
        )
    if x_overlap > 1.0e-7:
        candidates.extend(
            [
                (
                    abs(boxes[left, 1] + boxes[left, 3] - boxes[right, 1]),
                    1,
                    left,
                    right,
                ),
                (
                    abs(boxes[right, 1] + boxes[right, 3] - boxes[left, 1]),
                    1,
                    right,
                    left,
                ),
            ]
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            row[0],
            bool(instance.is_preplaced[row[3]]),
        ),
    )


def _contact_options(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    tolerance: float,
) -> list[tuple[float, int, int, int]]:
    output: list[tuple[float, int, int, int]] = []
    cluster = instance.cluster_id
    for cluster_id in np.unique(cluster[cluster > 0]):
        members = np.nonzero(cluster == cluster_id)[0]
        if len(members) < 2:
            continue
        labels = component_labels(boxes, members)
        for label in sorted(set(labels.values())):
            component = sorted(int(index) for index in members if labels[int(index)] == label)
            pair_options = [
                option
                for offset, left in enumerate(component)
                for right in component[offset + 1 :]
                if (
                    option := _pair_contact_option(
                        boxes,
                        instance,
                        left,
                        right,
                    )
                )
                is not None
            ]
            output.extend(option for option in pair_options if option[0] <= tolerance)
    return output


def _contact_groups(
    contacts: list[tuple[float, int, int, int]],
) -> list[tuple[list[int], list[tuple[float, int, int, int]]]]:
    adjacency: dict[int, set[int]] = {}
    for _gap, _axis, left, right in contacts:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    groups = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        node_set = set(nodes)
        groups.append(
            (
                sorted(nodes),
                [row for row in contacts if row[2] in node_set and row[3] in node_set],
            )
        )
    return groups


def _audit(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    resolution: float,
) -> tuple:
    report = check_feasibility(Placement(boxes, instance))
    soft = check_soft(Placement(boxes, instance))
    aligned = np.rint(boxes / resolution) * resolution
    residual = float(np.max(np.abs(boxes - aligned), initial=0.0))
    return report, soft, residual


def _solve_integer_projection(
    boxes: np.ndarray,
    continuous: np.ndarray,
    instance: FloorplanInstance,
    *,
    resolution: float,
    tight_threshold: float,
    contact_tolerance: float,
    time_limit_ms: int,
    contact_reference: np.ndarray | None = None,
) -> tuple[
    np.ndarray | None,
    str,
    list[tuple[float, int, int, int]],
    float,
]:
    count = len(boxes)
    solver = _solver(time_limit_ms)
    if solver is None:
        return None, "", [], 0.0
    origin = np.floor(continuous[:, :2].min(axis=0) / resolution) * resolution
    centered = continuous.copy()
    centered[:, :2] -= origin
    size_units = np.rint(boxes[:, 2:] / resolution).astype(np.int64)
    positions = np.rint(centered[:, :2] / resolution).astype(np.int64)
    span_units = max(
        float((centered[:, 0] + boxes[:, 2]).max()),
        float((centered[:, 1] + boxes[:, 3]).max()),
        1.0,
    )
    margin_units = int(np.ceil(max(contact_tolerance, 0.01 * span_units) / resolution))
    width_units = int(np.ceil((centered[:, 0] + boxes[:, 2]).max() / resolution)) + margin_units
    height_units = int(np.ceil((centered[:, 1] + boxes[:, 3]).max() / resolution)) + margin_units
    reference = continuous if contact_reference is None else contact_reference
    contacts = _contact_options(
        reference,
        instance,
        contact_tolerance,
    )
    contact_pairs = {(left, right) for _gap, _axis, left, right in contacts}
    x = [solver.IntVar(0, max(0, width_units - int(size_units[i, 0])), "") for i in range(count)]
    y = [solver.IntVar(0, max(0, height_units - int(size_units[i, 1])), "") for i in range(count)]
    x_min = solver.IntVar(0, width_units, "")
    y_min = solver.IntVar(0, height_units, "")
    x_max = solver.IntVar(0, width_units, "")
    y_max = solver.IntVar(0, height_units, "")
    objective = solver.Objective()
    objective.SetMinimization()
    for index in range(count):
        solver.Add(x[index] >= x_min)
        solver.Add(y[index] >= y_min)
        solver.Add(x[index] + size_units[index, 0] <= x_max)
        solver.Add(y[index] + size_units[index, 1] <= y_max)
        dx = solver.IntVar(0, width_units, "")
        dy = solver.IntVar(0, height_units, "")
        solver.Add(dx >= x[index] - int(positions[index, 0]))
        solver.Add(dx >= int(positions[index, 0]) - x[index])
        solver.Add(dy >= y[index] - int(positions[index, 1]))
        solver.Add(dy >= int(positions[index, 1]) - y[index])
        objective.SetCoefficient(dx, 1.0 / max(width_units, 1))
        objective.SetCoefficient(dy, 1.0 / max(height_units, 1))
        if instance.is_preplaced[index]:
            solver.Add(x[index] == int(positions[index, 0]))
            solver.Add(y[index] == int(positions[index, 1]))
    _, baseline_soft, _ = _audit(reference, instance, resolution)
    baseline_bad_boundary = set(baseline_soft.boundary_blocks)
    for index in np.nonzero(instance.boundary_code != 0)[0]:
        if int(index) in baseline_bad_boundary:
            continue
        code = int(instance.boundary_code[index])
        slacks = []
        if code & 1:
            slacks.append(x[index] - x_min)
        if code & 2:
            slacks.append(x_max - x[index] - size_units[index, 0])
        if code & 4:
            slacks.append(y_max - y[index] - size_units[index, 1])
        if code & 8:
            slacks.append(y[index] - y_min)
        for expression in slacks:
            slack = solver.IntVar(0, max(width_units, height_units), "")
            solver.Add(slack >= expression)
            objective.SetCoefficient(slack, 100.0)
    big_x = width_units * 2
    big_y = height_units * 2
    for left in range(count - 1):
        for right in range(left + 1, count):
            x_gap = max(
                continuous[right, 0] - continuous[left, 0] - continuous[left, 2],
                continuous[left, 0] - continuous[right, 0] - continuous[right, 2],
            )
            y_gap = max(
                continuous[right, 1] - continuous[left, 1] - continuous[left, 3],
                continuous[left, 1] - continuous[right, 1] - continuous[right, 3],
            )
            pair_fixed = (
                max(x_gap, y_gap) >= tight_threshold
                and (left, right) not in contact_pairs
                and (right, left) not in contact_pairs
            )
            if pair_fixed:
                axis, low, high = pair_relation(continuous, left, right)
                if axis == 0:
                    solver.Add(x[low] + size_units[low, 0] <= x[high])
                else:
                    solver.Add(y[low] + size_units[low, 1] <= y[high])
                continue
            before_x = solver.BoolVar("")
            after_x = solver.BoolVar("")
            before_y = solver.BoolVar("")
            after_y = solver.BoolVar("")
            solver.Add(x[left] + size_units[left, 0] <= x[right] + big_x * (1 - before_x))
            solver.Add(x[right] + size_units[right, 0] <= x[left] + big_x * (1 - after_x))
            solver.Add(y[left] + size_units[left, 1] <= y[right] + big_y * (1 - before_y))
            solver.Add(y[right] + size_units[right, 1] <= y[left] + big_y * (1 - after_y))
            solver.Add(before_x + after_x + before_y + after_y >= 1)
    selected_contacts: list[tuple[object, float, int, int, int]] = []
    tolerance_units = max(1, int(contact_tolerance / resolution))
    for nodes, edges in _contact_groups(contacts):
        root = nodes[0]
        outgoing = {node: [] for node in nodes}
        incoming = {node: [] for node in nodes}
        for original_gap, axis, low, high in edges:
            use = solver.BoolVar("")
            selected_contacts.append((use, original_gap, axis, low, high))
            gap = solver.IntVar(0, tolerance_units, "")
            if axis == 0:
                solver.Add(x[low] + size_units[low, 0] <= x[high] + big_x * (1 - use))
                solver.Add(gap >= x[high] - x[low] - size_units[low, 0] - big_x * (1 - use))
                solver.Add(y[low] + size_units[low, 1] >= y[high] + 1 - big_y * (1 - use))
                solver.Add(y[high] + size_units[high, 1] >= y[low] + 1 - big_y * (1 - use))
            else:
                solver.Add(y[low] + size_units[low, 1] <= y[high] + big_y * (1 - use))
                solver.Add(gap >= y[high] - y[low] - size_units[low, 1] - big_y * (1 - use))
                solver.Add(x[low] + size_units[low, 0] >= x[high] + 1 - big_x * (1 - use))
                solver.Add(x[high] + size_units[high, 0] >= x[low] + 1 - big_x * (1 - use))
            solver.Add(gap <= tolerance_units * (1 - use) + 1)
            objective.SetCoefficient(
                gap,
                100.0 / (1.0 + original_gap / max(contact_tolerance, resolution)),
            )
            for tail, head in ((low, high), (high, low)):
                flow = solver.NumVar(0, len(nodes), "")
                solver.Add(flow <= len(nodes) * use)
                outgoing[tail].append(flow)
                incoming[head].append(flow)
        for node in nodes:
            balance = -(len(nodes) - 1) if node == root else 1
            solver.Add(sum(outgoing[node]) - sum(incoming[node]) == balance)
    status = solver.Solve()
    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ):
        return None, "SCIP", [], float(solver.WallTime())
    projected = boxes.copy()
    projected[:, 0] = np.asarray([value.solution_value() for value in x]) * resolution + origin[0]
    projected[:, 1] = np.asarray([value.solution_value() for value in y]) * resolution + origin[1]
    projected[:, 2:] = size_units * resolution
    selected = [
        (original_gap, axis, low, high)
        for use, original_gap, axis, low, high in selected_contacts
        if use.solution_value() > 0.5
    ]
    return projected, "SCIP", selected, float(solver.WallTime())


def _snap_contacts(
    boxes: np.ndarray,
    contacts: list[tuple[float, int, int, int]],
    instance: FloorplanInstance,
) -> np.ndarray:
    output = boxes.copy()
    for _ in range(6):
        for _gap, axis, low, high in contacts:
            if instance.is_preplaced[high]:
                if instance.is_preplaced[low]:
                    continue
                target = output[high, axis] - output[low, axis + 2]
                if abs(float(output[low, axis] - target)) <= 1.0e-6:
                    output[low, axis] = target
                continue
            target = output[low, axis] + output[low, axis + 2]
            if abs(float(output[high, axis] - target)) <= 1.0e-6:
                output[high, axis] = target
    return output


def _snap_contact_tree(
    boxes: np.ndarray,
    contacts: list[tuple[float, int, int, int]],
    instance: FloorplanInstance,
) -> np.ndarray:
    output = boxes.copy()
    parent = {index: index for index in range(len(boxes))}
    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    tree = []
    for option in sorted(contacts):
        _gap, _axis, left, right = option
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            continue
        parent[left_root] = right_root
        tree.append(option)
    adjacency = {index: [] for index in range(len(boxes))}
    for _gap, axis, low, high in tree:
        adjacency[low].append((high, axis, True))
        adjacency[high].append((low, axis, False))
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited or not adjacency[start]:
            continue
        component = []
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _axis, _is_high in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        anchor = next(
            (index for index in component if instance.is_preplaced[index]),
            min(component),
        )
        assigned = {anchor}
        queue = [anchor]
        while queue:
            current = queue.pop(0)
            for child, axis, child_is_high in adjacency[current]:
                if child in assigned:
                    continue
                if child_is_high:
                    output[child, axis] = (
                        output[current, axis] + output[current, axis + 2] - 1.0e-7
                    )
                else:
                    output[child, axis] = (
                        output[current, axis] - output[child, axis + 2] + 1.0e-7
                    )
                assigned.add(child)
                queue.append(child)
    return output


def project_quantized_layout(
    boxes: np.ndarray,
    instance: FloorplanInstance,
    *,
    resolution: float = 0.001,
    tight_threshold: float = 0.25,
    contact_tolerance: float = 0.02,
    legalization_time_limit_ms: int = 5000,
    projection_time_limit_ms: int = 15000,
    contact_reference: np.ndarray | None = None,
    legalizer_relation: str = "expanded",
) -> DbuProjectionResult:
    incoming = np.asarray(boxes, dtype=np.float64)
    if incoming.shape != (instance.block_count, 4):
        raise ValueError("boxes must have shape (blocks, 4)")
    if resolution <= 0.0 or tight_threshold <= 0.0:
        raise ValueError("resolution and tight_threshold must be positive")
    if contact_tolerance < 0.0:
        raise ValueError("contact_tolerance must be non-negative")
    if not np.isfinite(incoming).all():
        raise ValueError("boxes must be finite")
    quantized_instance = _quantized_instance(incoming, instance)
    packing_input = incoming.copy()
    offset = (np.arange(len(incoming))[:, None] * [37, 53]) % [17, 19] - [8, 9]
    packing_input[:, :2] += 0.1 * resolution * offset / [8, 9]
    continuous = np.asarray(
        scale_pack_legalize_reproducible(
            Placement(packing_input, quantized_instance),
            time_limit_ms=legalization_time_limit_ms,
            reshape=False,
            relation=legalizer_relation,
        ).xywh,
        dtype=np.float64,
    )
    initial_report, initial_soft, _ = _audit(
        continuous,
        instance,
        resolution,
    )
    projected, solver, selected_contacts, solver_seconds = _solve_integer_projection(
        incoming,
        continuous,
        instance,
        resolution=resolution,
        tight_threshold=tight_threshold,
        contact_tolerance=contact_tolerance,
        time_limit_ms=projection_time_limit_ms,
        contact_reference=contact_reference,
    )
    if projected is None:
        final_report, final_soft, residual = _audit(
            incoming,
            instance,
            resolution,
        )
        return DbuProjectionResult(
            boxes=incoming.copy(),
            accepted=False,
            route="integer_projection_infeasible",
            solver=solver,
            continuous_boxes=continuous,
            overlap_before=len(initial_report.overlap_pairs),
            overlap_after=len(final_report.overlap_pairs),
            area_violations=len(final_report.area_violations),
            fixed_violations=len(final_report.fixed_violations),
            preplaced_violations=len(final_report.preplaced_violations),
            aspect_violations=len(final_report.aspect_violations),
            grouping_before=initial_soft.grouping,
            grouping_after=final_soft.grouping,
            boundary_before=initial_soft.boundary,
            boundary_after=final_soft.boundary,
            max_grid_residual=residual,
            max_displacement=0.0,
            moved_blocks=0,
            contact_edges=len(selected_contacts),
            solver_seconds=solver_seconds,
        )

    def acceptable(candidate: np.ndarray) -> bool:
        report, soft, residual = _audit(candidate, instance, resolution)
        return bool(
            report.is_feasible()
            and soft.grouping <= initial_soft.grouping
            and soft.boundary <= initial_soft.boundary
            and residual <= 1.0e-4
        )

    final = _snap_contacts(projected, selected_contacts, instance)
    accepted = acceptable(final)
    route = "scale_pack_integer_projection_sequential"
    contact_edges = len(selected_contacts)
    if not accepted:
        tree_contacts = _contact_options(
            continuous if contact_reference is None else contact_reference,
            instance,
            contact_tolerance,
        )
        final = _snap_contact_tree(projected, tree_contacts, instance)
        accepted = acceptable(final)
        route = "scale_pack_integer_projection_contact_tree"
        contact_edges = len(tree_contacts)
    if not accepted:
        final = incoming.copy()
        route = "rejected_after_dual_contact_routes"
    final_report, final_soft, final_residual = _audit(
        final,
        instance,
        resolution,
    )
    displacement = np.max(
        np.abs(final[:, :2] - incoming[:, :2]),
        initial=0.0,
    )
    return DbuProjectionResult(
        boxes=final,
        accepted=accepted,
        route=route,
        solver=solver,
        continuous_boxes=continuous,
        overlap_before=len(initial_report.overlap_pairs),
        overlap_after=len(final_report.overlap_pairs),
        area_violations=len(final_report.area_violations),
        fixed_violations=len(final_report.fixed_violations),
        preplaced_violations=len(final_report.preplaced_violations),
        aspect_violations=len(final_report.aspect_violations),
        grouping_before=initial_soft.grouping,
        grouping_after=final_soft.grouping,
        boundary_before=initial_soft.boundary,
        boundary_after=final_soft.boundary,
        max_grid_residual=final_residual,
        max_displacement=float(displacement),
        moved_blocks=int(np.count_nonzero(np.linalg.norm(final[:, :2] - incoming[:, :2], axis=1) > resolution)),
        contact_edges=contact_edges,
        solver_seconds=solver_seconds,
    )


__all__ = ["DbuProjectionResult", "project_quantized_layout"]
