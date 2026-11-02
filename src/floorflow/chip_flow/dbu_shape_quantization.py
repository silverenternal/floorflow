"""Adjacency- and boundary-aware soft-shape quantization for chip endpoints."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from trinity.floorplan.geometry import bounding_box
from trinity.floorplan.scoring.feasibility import check_feasibility
from trinity.floorplan.scoring.soft import check_boundary, check_soft
from trinity.floorplan.types import FloorplanInstance, Placement

from floorflow.chip_flow.dbu_projection import (
    _contact_groups,
    _contact_options,
    _snap_contact_tree,
)
from floorflow.chip_flow.dbu_quantization import quantize_boxes_to_dbu


@dataclass(frozen=True)
class DbuShapeQuantizationOutput:
    """DBU boxes plus raw-reference soft and hard diagnostics."""

    boxes: np.ndarray
    area_relative_error: np.ndarray
    area_violation_mask: np.ndarray
    grouping_before: int
    grouping_after: int
    boundary_before: int
    boundary_after: int
    overlap_pairs: int
    max_grid_residual: float
    fixed_shape_error: float
    preplaced_max_error: float
    modified_shapes: int
    translated_components: int
    bridged_blocks: int


def _grid(value: float, resolution: float) -> float:
    return round(value / resolution) * resolution


def _area_error(
    boxes: np.ndarray,
    areas: np.ndarray,
    soft: np.ndarray,
) -> np.ndarray:
    relative = np.abs(boxes[:, 2] * boxes[:, 3] - areas) / np.maximum(areas, 1.0e-12)
    return np.where(soft, relative, 0.0)


def _boundary_shape_pass(
    boxes: np.ndarray,
    areas: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
    resolution: float,
    area_tolerance: float,
) -> tuple[np.ndarray, int]:
    output = boxes.copy()
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    mib = constraints[:, 2].astype(int) != 0
    modified = 0
    for _ in range(3):
        before_grouping = check_soft(Placement(output, instance)).grouping
        before_boundary = check_boundary(Placement(output, instance))[0]
        x_min, y_min, x_max, y_max = bounding_box(output)
        for index in np.nonzero(constraints[:, 4] != 0)[0]:
            if preplaced[index] or mib[index]:
                continue
            code = int(constraints[index, 4])
            candidate = output.copy()
            if code & 1:
                candidate[index, 0] = x_min
            if code & 8:
                candidate[index, 1] = y_min
            if code & 2:
                if not fixed[index]:
                    candidate[index, 2] = x_max - candidate[index, 0]
                else:
                    candidate[index, 0] = x_max - candidate[index, 2]
            if code & 4:
                if not fixed[index]:
                    candidate[index, 3] = y_max - candidate[index, 1]
                else:
                    candidate[index, 1] = y_max - candidate[index, 3]
            if not fixed[index]:
                area = candidate[index, 2] * candidate[index, 3]
                if abs(area - areas[index]) / max(areas[index], 1.0e-12) > area_tolerance:
                    continue
            report = check_feasibility(Placement(candidate, instance))
            soft = check_soft(Placement(candidate, instance))
            if (
                not report.fixed_violations
                and not report.preplaced_violations
                and soft.grouping <= before_grouping
                and soft.boundary < before_boundary
            ):
                output = candidate
                before_boundary = soft.boundary
                modified += 1
    return output, modified


def _boundary_translation_pass(
    boxes: np.ndarray,
    contacts: list[tuple[float, int, int, int]],
    constraints: np.ndarray,
    instance: FloorplanInstance,
    resolution: float,
) -> tuple[np.ndarray, int]:
    output = boxes.copy()
    groups = [nodes for nodes, _edges in _contact_groups(contacts)]
    assigned = {index for group in groups for index in group}
    groups.extend([[index] for index in range(len(output)) if index not in assigned])
    preplaced = constraints[:, 1] != 0
    translated = 0
    for _ in range(10):
        changed = False
        for group in groups:
            if any(preplaced[index] for index in group):
                continue
            x_min, y_min, x_max, y_max = bounding_box(output)
            x_shifts = {0.0}
            y_shifts = {0.0}
            for index in group:
                code = int(constraints[index, 4])
                if code & 1:
                    x_shifts.add(_grid(x_min - output[index, 0], resolution))
                if code & 2:
                    x_shifts.add(
                        _grid(x_max - output[index, 0] - output[index, 2], resolution)
                    )
                if code & 8:
                    y_shifts.add(_grid(y_min - output[index, 1], resolution))
                if code & 4:
                    y_shifts.add(
                        _grid(y_max - output[index, 1] - output[index, 3], resolution)
                    )
            before = check_boundary(Placement(output, instance))[0]
            best = (before, 0.0, 0.0)
            for dx, dy in itertools.product(x_shifts, y_shifts):
                candidate = output.copy()
                candidate[group, 0] += dx
                candidate[group, 1] += dy
                boundary = check_boundary(Placement(candidate, instance))[0]
                if boundary < best[0]:
                    best = (boundary, dx, dy)
            if best[0] < before:
                output[group, 0] += best[1]
                output[group, 1] += best[2]
                translated += 1
                changed = True
        if not changed:
            break
    return output, translated


def _boundary_contact_bridge_pass(
    boxes: np.ndarray,
    constraints: np.ndarray,
    instance: FloorplanInstance,
) -> tuple[np.ndarray, int]:
    output = boxes.copy()
    preplaced = constraints[:, 1] != 0
    bridged = 0
    for _ in range(len(output)):
        before = check_soft(Placement(output, instance))
        if before.boundary == 0:
            break
        x_min, y_min, x_max, y_max = bounding_box(output)
        accepted = False
        for index in before.boundary_blocks:
            if preplaced[index]:
                continue
            code = int(constraints[index, 4])
            moves = []
            if code & 1:
                moves.append((0, x_min))
            if code & 2:
                moves.append((0, x_max - output[index, 2]))
            if code & 8:
                moves.append((1, y_min))
            if code & 4:
                moves.append((1, y_max - output[index, 3]))
            for axis, target in moves:
                delta = float(target - output[index, axis])
                for neighbor in range(len(output)):
                    if neighbor == index or preplaced[neighbor]:
                        continue
                    candidate = output.copy()
                    candidate[index, axis] = target
                    candidate[neighbor, axis] += delta
                    soft = check_soft(Placement(candidate, instance))
                    if (
                        soft.grouping <= before.grouping
                        and soft.boundary < before.boundary
                    ):
                        output = candidate
                        bridged += 1
                        accepted = True
                        break
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            break
    return output, bridged


def quantize_boxes_adjacency_aware(
    boxes: torch.Tensor,
    areas: torch.Tensor,
    constraints: torch.Tensor,
    instance: FloorplanInstance,
    *,
    resolution: float = 0.001,
    area_tolerance: float = 0.01,
    contact_tolerance: float = 0.02,
) -> DbuShapeQuantizationOutput:
    """Quantize shapes while restoring raw contact and boundary topology."""

    if boxes.ndim != 3 or boxes.shape[0] != 1:
        raise ValueError("boxes must have batch size one")
    if instance.block_count != boxes.shape[1]:
        raise ValueError("instance block count does not match boxes")
    initial = quantize_boxes_to_dbu(
        boxes.float(),
        areas.float(),
        constraints.float(),
        resolution=resolution,
        area_tolerance=area_tolerance,
    )
    raw = boxes[0].detach().cpu().numpy().astype(np.float64)
    source_areas = areas[0].detach().cpu().numpy().astype(np.float64)
    source_constraints = constraints[0].detach().cpu().numpy().astype(np.float64)
    quantized = initial.boxes[0].detach().cpu().numpy().astype(np.float64)
    contacts = _contact_options(raw, instance, contact_tolerance)
    contacted = _snap_contact_tree(quantized, contacts, instance)
    preplaced = source_constraints[:, 1] != 0
    if check_feasibility(Placement(contacted, instance)).preplaced_violations:
        contacted[preplaced] = quantized[preplaced]
    shaped, modified = _boundary_shape_pass(
        contacted,
        source_areas,
        source_constraints,
        instance,
        resolution,
        area_tolerance,
    )
    translated, translation_count = _boundary_translation_pass(
        shaped,
        contacts,
        source_constraints,
        instance,
        resolution,
    )
    bridged, bridge_count = _boundary_contact_bridge_pass(
        translated,
        source_constraints,
        instance,
    )
    report = check_feasibility(Placement(bridged, instance))
    soft = check_soft(Placement(bridged, instance))
    initial_soft = check_soft(Placement(raw, instance))
    aligned = np.rint(bridged / resolution) * resolution
    soft_mask = source_constraints[:, 0] == 0
    soft_mask &= source_constraints[:, 1] == 0
    area_relative = _area_error(bridged, source_areas, soft_mask)
    soft_grid_residual = np.abs(bridged[soft_mask] - aligned[soft_mask])
    return DbuShapeQuantizationOutput(
        boxes=bridged,
        area_relative_error=area_relative,
        area_violation_mask=soft_mask & (area_relative > area_tolerance),
        grouping_before=int(initial_soft.grouping),
        grouping_after=int(soft.grouping),
        boundary_before=int(initial_soft.boundary),
        boundary_after=int(soft.boundary),
        overlap_pairs=len(report.overlap_pairs),
        max_grid_residual=float(np.max(soft_grid_residual, initial=0.0)),
        fixed_shape_error=float(
            np.max(
                np.abs(bridged[source_constraints[:, 0] != 0, 2:] - raw[source_constraints[:, 0] != 0, 2:]),
                initial=0.0,
            )
        ),
        preplaced_max_error=float(
            np.max(
                np.abs(bridged[source_constraints[:, 1] != 0] - raw[source_constraints[:, 1] != 0]),
                initial=0.0,
            )
        ),
        modified_shapes=modified,
        translated_components=translation_count,
        bridged_blocks=bridge_count,
    )


__all__ = ["DbuShapeQuantizationOutput", "quantize_boxes_adjacency_aware"]
