"""Convert selected room-mask logits into canonical room polygon rows."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt
from scipy.ndimage import label as component_label
from shapely import set_precision
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import CanonicalOperation
from floorflow.geometry.owner_cell_complex import (
    OwnerCellComplex,
    extract_owner_cell_complex,
)
from floorflow.geometry.partition_polygons import component_cell_polygon

_CANONICAL_GRID_SIZE = 1.0 / 16.0
_CANONICAL_INSET = 0.1
_CRACK_STRUCTURE = np.ones((3, 3), dtype=bool)
_MAX_CRACK_HOLE_PIXELS = 16
_MAX_BRIDGE_GAP_PIXELS = 6.0


@dataclass(frozen=True)
class MaskPartitionReadout:
    rooms: list[dict]
    operations: list[CanonicalOperation]
    adjacency: list[list[str]] | None
    cell_complex: OwnerCellComplex | None


def _support_is_watertight(
    support: np.ndarray,
    minimum_component_area: int = 1,
) -> bool:
    if not support.any():
        return False
    holes = binary_fill_holes(support) & ~support
    components, _count = component_label(support)
    sizes = np.bincount(components.ravel())[1:]
    retained = int(
        np.count_nonzero(sizes >= max(1, int(minimum_component_area)))
    )
    return not bool(holes.any()) and retained <= 1


def fixed_point_safe_polygon(shape: Polygon) -> Polygon:
    """Snap to the token grid without carrying quantized self-intersections."""

    snapped = set_precision(shape, _CANONICAL_GRID_SIZE, mode="valid_output")
    if snapped.geom_type == "GeometryCollection":
        polygons = [part for part in snapped.geoms if part.geom_type == "Polygon"]
        snapped = max(polygons, key=lambda part: part.area) if polygons else Polygon()
    if snapped.geom_type == "MultiPolygon":
        snapped = max(snapped.geoms, key=lambda part: part.area)
    if snapped.geom_type != "Polygon":
        return Polygon()
    return snapped


def _largest_polygon_geometry(shape):
    """Return the largest polygonal member of a mixed Shapely result."""

    if shape.geom_type == "Polygon":
        return shape
    candidates = []
    if hasattr(shape, "geoms"):
        for part in shape.geoms:
            candidate = _largest_polygon_geometry(part)
            if candidate.geom_type == "Polygon" and not candidate.is_empty:
                candidates.append(candidate)
    return max(candidates, key=lambda item: item.area) if candidates else Polygon()


def _bounded_crack_repair(support: np.ndarray) -> np.ndarray:
    """Close cracks without growing the exterior support silhouette.

    A closing adds both bridges between support components and a one-pixel
    exterior rim.  Keep only small bounded holes and added components that
    touch at least two distinct original support components.
    """

    repaired = support.copy()
    holes = binary_fill_holes(support) & ~support
    hole_labels, hole_count = component_label(
        holes,
        structure=_CRACK_STRUCTURE,
    )
    for index in range(1, int(hole_count) + 1):
        component = hole_labels == index
        if int(component.sum()) <= _MAX_CRACK_HOLE_PIXELS:
            repaired |= component

    support_labels, _component_count = component_label(
        support,
        structure=_CRACK_STRUCTURE,
    )
    closed = binary_closing(support, structure=_CRACK_STRUCTURE)
    additions = closed & ~support
    addition_labels, addition_count = component_label(
        additions,
        structure=_CRACK_STRUCTURE,
    )
    for index in range(1, int(addition_count) + 1):
        component = addition_labels == index
        neighbours = np.zeros_like(support)
        neighbours[1:, :] |= component[:-1, :]
        neighbours[:-1, :] |= component[1:, :]
        neighbours[:, 1:] |= component[:, :-1]
        neighbours[:, :-1] |= component[:, 1:]
        touched = np.unique(support_labels[neighbours & support])
        if np.count_nonzero(touched) >= 2:
            repaired |= component
    return repaired


def _bridge_support_components(
    support: np.ndarray,
    minimum_component_area: int,
) -> np.ndarray:
    """Connect substantial support islands with one-pixel shortest bridges."""

    labels, count = component_label(support)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component_ids = [
        index
        for index in range(1, int(count) + 1)
        if sizes[index] >= max(1, int(minimum_component_area))
    ]
    if len(component_ids) <= 1:
        return support
    component_ids.sort(key=lambda index: int(sizes[index]), reverse=True)
    bridged = support.copy()
    connected_ids = [component_ids[0]]
    for component_id in component_ids[1:]:
        connected_mask = np.isin(labels, connected_ids)
        distances, indices = distance_transform_edt(
            ~connected_mask,
            return_indices=True,
        )
        pixels = np.argwhere(labels == component_id)
        pixel_distances = distances[pixels[:, 0], pixels[:, 1]]
        target = pixels[int(np.argmin(pixel_distances))]
        if float(pixel_distances.min()) > _MAX_BRIDGE_GAP_PIXELS:
            continue
        source = np.asarray(
            [indices[0][tuple(target)], indices[1][tuple(target)]],
            dtype=np.int32,
        )
        cv2.line(
            bridged,
            (int(source[1]), int(source[0])),
            (int(target[1]), int(target[0])),
            1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        connected_ids.append(component_id)
    return bridged


def _apply_gap_policy(
    owner: np.ndarray,
    support: np.ndarray,
    footprint: np.ndarray,
    selected_count: int,
    policy: str,
    minimum_component_area: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    owner = owner.copy()
    support = support.copy()
    if policy == "threshold" or not support.any():
        return owner, support
    if policy in {"hole_fill", "bridge_fill"}:
        gaps = binary_fill_holes(support) & ~support
    else:
        gaps = footprint & ~support
    if gaps.any():
        _distances, indices = distance_transform_edt(
            ~support,
            return_indices=True,
        )
        owner[gaps] = owner[indices[0][gaps], indices[1][gaps]]
    support = (
        footprint
        if policy in {"nearest_fill", "partition_fill"}
        else support | gaps
    )
    if policy == "partition_fill":
        primary = np.zeros_like(support, dtype=bool)
        for local_index in range(selected_count):
            components, count = component_label(
                (owner == local_index) & support
            )
            if count == 0:
                continue
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            primary |= components == int(np.argmax(sizes))
        remaining = support & ~primary
        if remaining.any() and primary.any():
            _distances, indices = distance_transform_edt(
                ~primary,
                return_indices=True,
            )
            owner[remaining] = owner[
                indices[0][remaining],
                indices[1][remaining],
            ]
    elif policy == "crack_fill":
        closed = _bounded_crack_repair(support)
        closed &= footprint
        added = closed & ~support
        if added.any() and support.any():
            _distances, indices = distance_transform_edt(
                ~support,
                return_indices=True,
            )
            owner[added] = owner[
                indices[0][added],
                indices[1][added],
            ]
        support = closed
    elif policy == "bridge_fill":
        bridged = _bridge_support_components(
            support,
            minimum_component_area,
        )
        added = bridged & ~support
        if added.any() and support.any():
            _distances, indices = distance_transform_edt(
                ~support,
                return_indices=True,
            )
            owner[added] = owner[indices[0][added], indices[1][added]]
        support = bridged
    return owner, support


def _complete_joint_partition_owner(
    owner: np.ndarray,
    support: np.ndarray,
    selected_count: int,
    minimum_area: float,
) -> tuple[np.ndarray, list[int]]:
    """Fill components dropped by one-primary filtering from the owner partition."""

    primary_ids = []
    for _iteration in range(8):
        primary_masks = []
        primary_ids = []
        for local_index in range(selected_count):
            components, _count = component_label(
                (owner == local_index) & support
            )
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            primary = int(np.argmax(sizes))
            if primary > 0 and sizes[primary] >= max(1, int(minimum_area)):
                primary_masks.append(components == primary)
                primary_ids.append(local_index)
        if not primary_masks:
            return owner, []
        primary_union = np.logical_or.reduce(primary_masks)
        residual = support & ~primary_union
        if not residual.any():
            break
        _distances, indices = distance_transform_edt(
            ~primary_union,
            return_indices=True,
        )
        owner[residual] = owner[
            indices[0][residual],
            indices[1][residual],
        ]
    return owner, primary_ids


def decode_mask_partition(
    mask_logits: torch.Tensor,
    boundary_points: np.ndarray,
    selected: list[int],
    classes: np.ndarray,
    exists: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    canvas_size: int,
    confidence_threshold: float,
    minimum_area: float,
    gap_policy: str = "threshold",
    mask_probabilities: torch.Tensor | None = None,
    partition_support: str = "full_footprint",
) -> MaskPartitionReadout:
    """Decode one sample by max competition between selected room masks."""

    independent_probability = mask_logits[0, selected].sigmoid()
    probability = (
        mask_probabilities[0, selected]
        if mask_probabilities is not None
        else independent_probability
    )
    independent_probability = F.interpolate(
        independent_probability[None],
        size=(canvas_size, canvas_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    probability = F.interpolate(
        probability[None],
        size=(canvas_size, canvas_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    footprint_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(
        footprint_mask,
        [np.rint(boundary_points).astype(np.int32)],
        1,
    )
    owner = probability.argmax(dim=0).cpu().numpy()
    confidence = probability.max(dim=0).values.cpu().numpy()
    if partition_support not in {"full_footprint", "independent_confidence"}:
        raise ValueError("unknown partition support policy")
    support = footprint_mask.astype(bool)
    if mask_probabilities is None:
        support &= confidence >= float(confidence_threshold)
    elif partition_support == "independent_confidence":
        support &= (
            independent_probability.max(dim=0).values.cpu().numpy()
            >= float(confidence_threshold)
        )
    if gap_policy not in {
        "threshold",
        "hole_fill",
        "nearest_fill",
        "partition_fill",
        "crack_fill",
        "bridge_fill",
        "adaptive_fill",
    }:
        raise ValueError("unknown mask partition gap policy")
    chosen_policy = gap_policy
    if gap_policy == "adaptive_fill":
        for candidate in (
            "threshold",
            "hole_fill",
            "bridge_fill",
            "partition_fill",
        ):
            candidate_owner, candidate_support = _apply_gap_policy(
                owner,
                support,
                footprint_mask.astype(bool),
                len(selected),
                candidate,
                minimum_area,
            )
            if _support_is_watertight(
                candidate_support,
                minimum_area,
            ):
                owner = candidate_owner
                support = candidate_support
                chosen_policy = candidate
                break
        else:
            chosen_policy = "partition_fill"
            owner, support = _apply_gap_policy(
                owner,
                support,
                footprint_mask.astype(bool),
                len(selected),
                chosen_policy,
                minimum_area,
            )
    else:
        owner, support = _apply_gap_policy(
            owner,
            support,
            footprint_mask.astype(bool),
            len(selected),
            gap_policy,
            minimum_area,
        )
    accepted = []
    accepted_local_indices = []
    operations = []
    if mask_probabilities is not None:
        owner, _primary_ids = _complete_joint_partition_owner(
            owner,
            support,
            len(selected),
            minimum_area,
        )
        operations.append(
            CanonicalOperation(
                "partition_owner_completion",
                "footprint",
            )
        )
    footprint = Polygon(boundary_points)
    precision_footprint = footprint.buffer(-_CANONICAL_INSET)
    if precision_footprint.is_empty:
        precision_footprint = footprint
    coordinate_scale = max(1, canvas_size - 1) / max(1, canvas_size)
    confidence_value = float(exists[selected].max())
    for local_index, query in enumerate(selected):
        components, _count = component_label(
            (owner == local_index) & support
        )
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        primary = int(np.argmax(sizes))
        component_indices = (
            [primary]
            if primary > 0 and sizes[primary] >= max(1, int(minimum_area))
            else []
        )
        for dropped in np.flatnonzero(sizes):
            if int(dropped) != primary:
                operations.append(
                    CanonicalOperation(
                        "room_drop_secondary_query_component",
                        f"query_{query:02d}",
                    )
                )
        for component_index in component_indices:
            component = components == component_index
            if int(component.sum()) < max(1, int(minimum_area)):
                operations.append(
                    CanonicalOperation(
                        "room_drop_mask_fragment",
                        f"query_{query:02d}",
                    )
                )
                continue
            shape = component_cell_polygon(component)
            shape = Polygon(
                np.asarray(shape.exterior.coords[:-1]) * coordinate_scale,
                [
                    np.asarray(hole.coords[:-1]) * coordinate_scale
                    for hole in shape.interiors
                ],
            )
            unsnapped = shape
            shape = fixed_point_safe_polygon(shape)
            if not shape.equals(unsnapped):
                operations.append(
                    CanonicalOperation(
                        "room_snap_canonical_grid",
                        f"query_{query:02d}",
                    )
                )
            clipped = shape.intersection(precision_footprint)
            if not precision_footprint.equals(footprint):
                operations.append(
                    CanonicalOperation(
                        "room_clip_precision_guard",
                        f"query_{query:02d}",
                    )
                )
            if clipped.geom_type == "MultiPolygon" or (
                mask_probabilities is not None
                and clipped.geom_type == "GeometryCollection"
            ):
                clipped = _largest_polygon_geometry(clipped)
                operations.append(
                    CanonicalOperation(
                        "room_keep_largest_mask_clip",
                        f"query_{query:02d}",
                    )
                )
            if (
                clipped.geom_type != "Polygon"
                or clipped.is_empty
                or clipped.area < float(minimum_area)
            ):
                operations.append(
                    CanonicalOperation(
                        "room_drop_mask_fragment",
                        f"query_{query:02d}",
                    )
                )
                continue
            class_index = int(classes[query])
            accepted.append(
                {
                    "id": f"room_{len(accepted) + 1:02d}",
                    "label": taxonomy.class_names[
                        taxonomy.room_ids[class_index]
                    ],
                    "polygon": np.asarray(
                        clipped.exterior.coords[:-1]
                    ).tolist(),
                    "holes": [
                        np.asarray(hole.coords[:-1]).tolist()
                        for hole in clipped.interiors
                    ],
                    "confidence": confidence_value,
                }
            )
            accepted_local_indices.append(local_index)
    operations.append(
        CanonicalOperation(
            "mask_partition_readout",
            "all",
            f"selected={len(selected)};accepted={len(accepted)}",
        )
    )
    if chosen_policy != "threshold":
        operations.append(
            CanonicalOperation(f"mask_{chosen_policy}", "footprint")
        )
    adjacency = None
    cell_complex = None
    if mask_probabilities is not None:
        local_to_id = {
            local_index: room["id"]
            for local_index, room in zip(
                accepted_local_indices,
                accepted,
                strict=True,
            )
        }
        cell_complex = extract_owner_cell_complex(
            owner,
            support,
            local_to_id,
            coordinate_scale=coordinate_scale,
            owner_confidence=independent_probability.max(
                dim=0
            ).values.cpu().numpy(),
            junction_tolerance=max(1.0, coordinate_scale),
        )
        adjacency = [list(edge) for edge in cell_complex.adjacency]
    return MaskPartitionReadout(
        rooms=accepted,
        operations=operations,
        adjacency=adjacency,
        cell_complex=cell_complex,
    )


def decode_mask_partition_rooms(
    *args,
    **kwargs,
) -> tuple[list[dict], list[CanonicalOperation], list[list[str]] | None]:
    """Compatibility wrapper returning the legacy three-value readout."""

    readout = decode_mask_partition(*args, **kwargs)
    return readout.rooms, readout.operations, readout.adjacency


__all__ = [
    "MaskPartitionReadout",
    "decode_mask_partition",
    "decode_mask_partition_rooms",
    "fixed_point_safe_polygon",
]
