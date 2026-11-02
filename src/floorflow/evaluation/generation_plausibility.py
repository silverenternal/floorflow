"""Target-free plausibility checks for generated floorplan rasters and plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as component_label
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    validity_checks,
    watertight_diagnostics,
)
from floorflow.evaluation.fpeval_bridge import to_fpeval_plan, verify
from floorflow.evaluation.structural_legality import wall_junction_summary
from floorflow.geometry.core import footprint_band, interior_mask
from floorflow.geometry.partition_polygons import explicit_partition_plan
from floorflow.geometry.reconciliation import reconcile_plan
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan


@dataclass(frozen=True)
class PlausibilityThresholds:
    """Frozen target-free thresholds for mechanism screening."""

    room_coverage_min: float = 0.90
    background_interior_max: float = 0.10
    unknown_interior_max: float = 0.01
    wall_closure_min: float = 0.95
    opening_alignment_min: float = 0.90
    furniture_containment_min: float = 0.90
    wall_continuity_min: float = 0.95
    room_face_retention_min: float = 0.80
    exportable_room_min_area_px: int = 32
    outside_footprint_max: float = 0.0


def _largest_component_fraction(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return 1.0
    components, count = component_label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 1.0
    sizes = np.bincount(components.ravel())[1:]
    return float(sizes.max() / binary.sum())


def _componentwise_largest_fraction(
    mask: np.ndarray,
    footprint_components: np.ndarray,
    component_count: int,
) -> float:
    structure = np.asarray(mask, dtype=bool) & (footprint_components > 0)
    total = int(structure.sum())
    if total == 0:
        return 1.0
    retained = 0
    for component_id in range(1, int(component_count) + 1):
        local = structure & (footprint_components == component_id)
        if not local.any():
            continue
        components, count = component_label(
            local,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())[1:]
        retained += int(sizes.max())
    return float(retained / total)


def _room_component_count(
    labels: np.ndarray,
    room_ids: np.ndarray,
    *,
    minimum_area: int = 1,
) -> int:
    components = 0
    for class_id in room_ids:
        component_map, count = component_label(labels == int(class_id))
        if count:
            sizes = np.bincount(component_map.ravel())[1:]
            components += int((sizes >= int(minimum_area)).sum())
    return components


def _plan_geometry_metrics(plan) -> dict[str, float]:
    polygons = []
    raw_validity = []
    for room in plan.rooms:
        coordinates = room.get("polygon") or []
        raw = (
            Polygon(coordinates, room.get("holes", []))
            if len(coordinates) >= 3
            else Polygon()
        )
        raw_validity.append(raw.is_valid)
        polygons.append(raw.buffer(0) if not raw.is_empty else Polygon())
    overlap = 0.0
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            overlap += polygons[left].intersection(polygons[right]).area
    union = unary_union(polygons) if polygons else Polygon()
    union_area = float(union.area) if not union.is_empty else 0.0
    frame = Polygon(
        ((0, 0), (plan.frame, 0), (plan.frame, plan.frame), (0, plan.frame))
    )
    outside = sum(polygon.difference(frame).area for polygon in polygons)
    components = (
        list(union.geoms)
        if hasattr(union, "geoms")
        else ([union] if not union.is_empty else [])
    )
    return {
        "room_closure_rate": (
            float(np.mean([polygon.is_valid for polygon in polygons]))
            if polygons
            else 0.0
        ),
        "self_intersection_rate": (
            1.0 - float(np.mean(raw_validity))
            if raw_validity
            else 1.0
        ),
        "room_overlap_ratio": (
            float(overlap / union_area) if union_area > 0.0 else 0.0
        ),
        "out_of_boundary_ratio": (
            float(outside / union_area) if union_area > 0.0 else 0.0
        ),
        "room_union_component_count": len(components),
        "room_union_area": union_area,
    }


def _raster_intrinsic_metrics(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    boundary_width: int = 7,
    support_radius: int = 2,
) -> dict[str, float | int]:
    labels = np.squeeze(np.asarray(labels, dtype=np.int64))
    footprint_mask = np.squeeze(np.asarray(footprint, dtype=bool))
    if footprint_mask.shape != labels.shape and footprint_mask.size == labels.size:
        footprint_mask = footprint_mask.reshape(labels.shape)
    footprint_components, footprint_component_count = component_label(footprint_mask)
    footprint_sizes = np.bincount(footprint_components.ravel())
    exportable_footprint_ids = np.flatnonzero(
        footprint_sizes >= 32
    )
    exportable_footprint_ids = exportable_footprint_ids[
        exportable_footprint_ids != 0
    ]
    exportable_footprint_mask = np.isin(
        footprint_components,
        exportable_footprint_ids,
    )
    footprint_tensor = torch.from_numpy(footprint_mask.astype(np.float32))[None, None]
    interior = interior_mask(footprint_tensor, boundary_width)[0, 0].numpy() > 0.5
    envelope = footprint_band(footprint_tensor, boundary_width)[0, 0].numpy() > 0.5
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    opening_ids = np.asarray(taxonomy.opening_ids, dtype=np.int64)
    furniture_ids = np.asarray(taxonomy.furniture_ids, dtype=np.int64)
    room_mask = np.isin(labels, room_ids) & footprint_mask
    opening_mask = np.isin(labels, opening_ids) & footprint_mask
    furniture_mask = np.isin(labels, furniture_ids) & footprint_mask
    wall_mask = labels == taxonomy.class_id("wall")
    structure_mask = wall_mask | opening_mask
    habitable = interior & ~structure_mask & ~furniture_mask

    wall_tensor = torch.from_numpy(wall_mask.astype(np.float32))[None, None]
    radius = max(1, int(support_radius))
    wall_support = F.max_pool2d(
        wall_tensor,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[0, 0].numpy() > 0.5
    wall_support |= envelope
    room_support = torch.from_numpy(binary_fill_holes(room_mask).astype(np.float32))[None, None]
    room_support = F.max_pool2d(
        room_support,
        kernel_size=25,
        stride=1,
        padding=12,
    )[0, 0].numpy() > 0.5

    habitable_count = max(1, int(habitable.sum()))
    opening_count = int(opening_mask.sum())
    furniture_count = int(furniture_mask.sum())
    structure_count = max(1, int(structure_mask.sum()))
    recognized = (
        room_mask
        | opening_mask
        | furniture_mask
        | wall_mask
        | (labels == 0)
    )
    room_components = _room_component_count(labels, room_ids)
    exportable_components = _room_component_count(
        labels,
        room_ids,
        minimum_area=32,
    )
    return {
        "room_coverage": float((room_mask & habitable).sum() / habitable_count),
        "room_area_fraction_of_interior": float(
            (room_mask & interior).sum() / max(1, int(interior.sum()))
        ),
        "background_interior_rate": float(
            ((labels == 0) & habitable).sum() / habitable_count
        ),
        "unknown_interior_rate": float(
            (~recognized & habitable).sum() / habitable_count
        ),
        "wall_area_fraction": float(
            (wall_mask & footprint_mask).sum() / max(1, footprint_mask.sum())
        ),
        "wall_closure": _largest_component_fraction(structure_mask),
        "componentwise_wall_closure": _componentwise_largest_fraction(
            structure_mask,
            footprint_components,
            footprint_component_count,
        ),
        "opening_wall_alignment": (
            float((opening_mask & wall_support).sum() / opening_count)
            if opening_count
            else 1.0
        ),
        "furniture_room_containment": (
            float((furniture_mask & room_support).sum() / furniture_count)
            if furniture_count
            else 1.0
        ),
        "outside_footprint_fraction": float(
            ((labels != 0) & ~footprint_mask).sum() / max(1, int((~footprint_mask).sum()))
        ),
        "footprint_component_count": int(footprint_component_count),
        "exportable_footprint_component_count": len(exportable_footprint_ids),
        "footprint_area_px": int(footprint_mask.sum()),
        "exportable_footprint_area_px": int(
            exportable_footprint_mask.sum()
        ),
        "room_component_count": int(room_components),
        "exportable_room_component_count": int(exportable_components),
        "tiny_room_component_rate": float(
            1.0 - exportable_components / max(1, room_components)
        ),
        "room_class_count": int(
            sum(bool(np.any(labels == int(class_id))) for class_id in room_ids)
        ),
        "wall_pixel_fraction_of_structure": float(wall_mask.sum() / structure_count),
    }


def raster_generation_plausibility(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    thresholds: PlausibilityThresholds | None = None,
    boundary_width: int = 7,
) -> dict[str, float | int | bool | list[str]]:
    """Measure target-free plausibility without reading reference geometry."""

    thresholds = thresholds or PlausibilityThresholds()
    metrics = _raster_intrinsic_metrics(
        labels,
        footprint,
        taxonomy,
        boundary_width=boundary_width,
    )
    checks = {
        "room_coverage_ok": metrics["room_coverage"] >= thresholds.room_coverage_min,
        "background_interior_ok": (
            metrics["background_interior_rate"] <= thresholds.background_interior_max
        ),
        "unknown_interior_ok": (
            metrics["unknown_interior_rate"] <= thresholds.unknown_interior_max
        ),
        "componentwise_wall_closure_ok": (
            metrics["componentwise_wall_closure"]
            >= thresholds.wall_closure_min
        ),
        "opening_alignment_ok": (
            metrics["opening_wall_alignment"] >= thresholds.opening_alignment_min
        ),
        "furniture_containment_ok": (
            metrics["furniture_room_containment"]
            >= thresholds.furniture_containment_min
        ),
        "outside_footprint_ok": (
            metrics["outside_footprint_fraction"] <= thresholds.outside_footprint_max
        ),
    }
    return {
        **metrics,
        **{f"raster_{name}": bool(value) for name, value in checks.items()},
        "raster_plausible": bool(all(checks.values())),
        "raster_failure_codes": [
            name.removesuffix("_ok") for name, value in checks.items() if not value
        ],
    }


def plan_generation_plausibility(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    thresholds: PlausibilityThresholds | None = None,
    tolerances: FEVPTolerances | None = None,
    explicit_rooms: bool = False,
    reconcile_gap_fraction: float | None = None,
    reconcile_use_ip: bool = False,
    repair_open_endpoints: bool = False,
) -> dict[str, float | int | bool | list[str]]:
    """Measure parseability, legality, topology, and junction plausibility."""

    thresholds = thresholds or PlausibilityThresholds()
    tolerances = tolerances or FEVPTolerances()
    labels = np.squeeze(np.asarray(labels, dtype=np.int64))
    footprint_mask = np.squeeze(np.asarray(footprint, dtype=bool))
    if footprint_mask.shape != labels.shape and footprint_mask.size == labels.size:
        footprint_mask = footprint_mask.reshape(labels.shape)
    footprint_components, footprint_component_count = component_label(
        footprint_mask
    )
    footprint_sizes = np.bincount(footprint_components.ravel())
    exportable_footprint_ids = np.flatnonzero(
        footprint_sizes >= thresholds.exportable_room_min_area_px
    )
    exportable_footprint_ids = exportable_footprint_ids[
        exportable_footprint_ids != 0
    ]
    exportable_footprint_mask = np.isin(
        footprint_components,
        exportable_footprint_ids,
    )
    plan = (
        explicit_partition_plan(labels, footprint_mask, taxonomy)
        if explicit_rooms
        else raster_labels_to_wall_plan(labels, taxonomy)
    )
    if reconcile_gap_fraction is not None:
        rooms = plan.rooms
        plan = reconcile_plan(
            plan,
            labels,
            taxonomy,
            gap_tolerance_fraction=float(reconcile_gap_fraction),
            use_ip=bool(reconcile_use_ip),
            repair_open_endpoints=bool(repair_open_endpoints),
        )
        plan.rooms = rooms
        plan.room_source = f"{plan.room_source}_walls_reconciled"
    geometry = _plan_geometry_metrics(plan)
    junction = wall_junction_summary(
        plan.walls,
        tolerance=tolerances.wall_junction_cluster_px,
    )
    validity = validity_checks(
        plan,
        footprint_mask,
        taxonomy,
        tolerances=tolerances,
        room_labels=labels,
    )
    fpeval_result = verify(to_fpeval_plan(plan))
    watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        tolerances=tolerances,
        fpeval_watertight=fpeval_result.watertight,
    )
    exportable_components = _room_component_count(
        labels,
        np.asarray(taxonomy.room_ids, dtype=np.int64),
        minimum_area=thresholds.exportable_room_min_area_px,
    )
    room_face_retention = (
        min(1.0, len(plan.rooms) / exportable_components)
        if exportable_components
        else 1.0
    )
    functional_watertight = bool(
        validity["validity"] == 1.0
        and geometry["room_overlap_ratio"] == 0.0
        and geometry["out_of_boundary_ratio"] == 0.0
        and geometry["room_union_area"]
        >= 0.999 * exportable_footprint_mask.sum()
        and geometry["room_union_component_count"]
        == len(exportable_footprint_ids)
        and room_face_retention >= thresholds.room_face_retention_min
    )
    wall_continuity_rate = float(junction["wall_continuity_rate"])
    wall_continuity_ok = wall_continuity_rate >= thresholds.wall_continuity_min
    face_retention_ok = (
        room_face_retention >= thresholds.room_face_retention_min
    )
    checks = {
        "parsable": bool(plan.rooms and plan.walls),
        "validity": float(validity["validity"]) == 1.0,
        "room_closure": float(geometry["room_closure_rate"]) == 1.0,
        "no_self_intersection": float(geometry["self_intersection_rate"]) == 0.0,
        "no_room_overlap": float(geometry["room_overlap_ratio"]) == 0.0,
        "out_of_boundary": float(geometry["out_of_boundary_ratio"]) == 0.0,
        "wall_continuity": bool(wall_continuity_ok),
        "room_face_retention": bool(face_retention_ok),
        "componentwise_watertight": functional_watertight,
        "functional_watertight": functional_watertight,
    }
    failure_codes = [str(code) for code in validity["validity_errors"]]
    failure_codes.extend(
        str(code) for code in watertight["watertight_issue_codes"]
    )
    failure_codes.extend(str(code) for code in fpeval_result.issues)
    if not wall_continuity_ok:
        failure_codes.append("wall_continuity")
    if not face_retention_ok:
        failure_codes.append("room_face_collapse")
    if fpeval_result.watertight and not functional_watertight:
        failure_codes.append("nonfunctional_watertight")
    elif len(exportable_footprint_ids) > 1:
        failure_codes.append("multi_component_footprint_protocol_boundary")
    return {
        "parsable": float(bool(plan.rooms and plan.walls)),
        "validity": float(validity["validity"]),
        "fpeval_validity": float(fpeval_result.valid),
        "watertight": float(fpeval_result.watertight),
        "room_face_retention": float(room_face_retention),
        "functional_watertight": float(functional_watertight),
        "watertight_diagnostic_rate": float(watertight["watertight_rate"]),
        "outer_boundary_closed": float(validity["outer_boundary_closed"]),
        "room_polygon_closed": float(validity["room_polygon_closed"]),
        "no_self_intersection": float(validity["no_self_intersection"]),
        "no_room_overlap": float(validity["no_room_overlap"]),
        "semantic_complete": float(validity["semantic_complete"]),
        "room_closure_rate": geometry["room_closure_rate"],
        "self_intersection_rate": geometry["self_intersection_rate"],
        "room_overlap_ratio": geometry["room_overlap_ratio"],
        "out_of_boundary_ratio": geometry["out_of_boundary_ratio"],
        "wall_continuity_rate": wall_continuity_rate,
        "hole_count": int(watertight["hole_count"]),
        "dangling_endpoint_rate": float(watertight["dangling_endpoint_rate"]),
        "wall_count": len(plan.walls),
        "room_face_count": len(plan.rooms),
        "footprint_component_count": int(footprint_component_count),
        "exportable_footprint_component_count": len(exportable_footprint_ids),
        "room_union_component_count": int(geometry["room_union_component_count"]),
        "room_union_area": float(geometry["room_union_area"]),
        "raster_exportable_room_component_count": int(exportable_components),
        "opening_count": len(plan.openings),
        **{f"plan_{name}_ok": bool(value) for name, value in checks.items()},
        "plan_plausible": bool(all(checks.values())),
        "plan_failure_codes": sorted(set(failure_codes)),
    }


def generation_plausibility(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    thresholds: PlausibilityThresholds | None = None,
    tolerances: FEVPTolerances | None = None,
    boundary_width: int = 7,
    explicit_rooms: bool = False,
    reconcile_gap_fraction: float | None = None,
    reconcile_use_ip: bool = False,
    repair_open_endpoints: bool = False,
) -> dict[str, float | int | bool | list[str]]:
    """Return both raster-level and CAD-level target-free plausibility."""

    thresholds = thresholds or PlausibilityThresholds()
    tolerances = tolerances or FEVPTolerances()
    return {
        **raster_generation_plausibility(
            labels,
            footprint,
            taxonomy,
            thresholds=thresholds,
            boundary_width=boundary_width,
        ),
        **plan_generation_plausibility(
            labels,
            footprint,
            taxonomy,
            thresholds=thresholds,
            tolerances=tolerances,
            explicit_rooms=explicit_rooms,
            reconcile_gap_fraction=reconcile_gap_fraction,
            reconcile_use_ip=reconcile_use_ip,
            repair_open_endpoints=repair_open_endpoints,
        ),
    }


__all__ = [
    "PlausibilityThresholds",
    "generation_plausibility",
    "plan_generation_plausibility",
    "raster_generation_plausibility",
]
