"""Shared room, boundary, and junction cell complex from an owner raster."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.ndimage import label as component_label
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union

from floorflow.geometry.partition_polygons import component_cell_polygon
from floorflow.geometry.wall_graph_readout import ReadoutWall


@dataclass(frozen=True)
class SharedBoundaryEdge:
    """One 1-cell separating two owners or an owner and the exterior."""

    id: str
    start: tuple[float, float]
    end: tuple[float, float]
    owner_a: int
    owner_b: int
    edge_type: str
    confidence: float

    @property
    def owner_pair(self) -> tuple[int, int]:
        return tuple(sorted((self.owner_a, self.owner_b)))

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True)
class OwnerJunction:
    """One 0-cell incidence cluster over shared boundary endpoints."""

    id: str
    point: tuple[float, float]
    degree: int
    boundary_ids: tuple[str, ...]


@dataclass(frozen=True)
class OwnerCellComplex:
    """Explicit 2-cell, 1-cell, and 0-cell topology from room ownership."""

    owner: np.ndarray
    support: np.ndarray
    coordinate_scale: float
    room_id_by_owner: tuple[tuple[int, str], ...]
    boundaries: tuple[SharedBoundaryEdge, ...]
    adjacency: tuple[tuple[str, str], ...]
    junctions: tuple[OwnerJunction, ...]

    def room_id(self, owner_index: int) -> str:
        mapping = dict(self.room_id_by_owner)
        if owner_index not in mapping:
            raise KeyError(f"owner {owner_index} has no canonical room")
        return mapping[owner_index]

    def wall_candidates(
        self,
        *,
        minimum_length: float = 2.5,
        thickness: float = 3.0,
        include_void: bool = True,
    ) -> tuple[ReadoutWall, ...]:
        """Convert shared boundaries into deterministic wall candidates."""

        return tuple(
            ReadoutWall(
                start=edge.start,
                end=edge.end,
                thickness=float(thickness),
                wall_type=(
                    "exterior" if edge.edge_type == "void" else "interior"
                ),
                continuity=float(edge.confidence),
            )
            for edge in self.boundaries
            if edge.length >= float(minimum_length)
            and (include_void or edge.edge_type == "interior")
        )

    def room_polygons(self) -> dict[str, Polygon]:
        """Recover exact owner-cell polygons without independent snapping."""

        polygons = {}
        for owner_index, room_id in self.room_id_by_owner:
            components, count = component_label(
                self.support & (self.owner == owner_index)
            )
            if count == 0:
                continue
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            primary = int(np.argmax(sizes))
            if primary == 0:
                continue
            polygon = component_cell_polygon(components == primary)
            coordinates = np.asarray(
                polygon.exterior.coords[:-1],
                dtype=np.float64,
            ) * self.coordinate_scale
            holes = [
                np.asarray(hole.coords[:-1], dtype=np.float64)
                * self.coordinate_scale
                for hole in polygon.interiors
            ]
            polygons[room_id] = Polygon(coordinates, holes)
        return polygons

    def merged_wall_candidates(
        self,
        *,
        simplify_epsilon: float = 1.0,
        minimum_length: float = 2.01,
        thickness: float = 3.0,
    ) -> tuple[ReadoutWall, ...]:
        """Merge owner 1-cells into continuous, simplified wall polylines."""

        network = linemerge(
            unary_union(
                [
                    LineString([edge.start, edge.end])
                    for edge in self.boundaries
                    if edge.length > 0.0
                ]
            )
        )
        lines = (
            list(network.geoms)
            if hasattr(network, "geoms")
            else [network]
        )
        candidates = []
        for line in lines:
            if line.is_empty or line.geom_type != "LineString":
                continue
            simplified = line.simplify(
                float(simplify_epsilon),
                preserve_topology=False,
            )
            coordinates = list(simplified.coords)
            for start, end in pairwise(coordinates):
                if math.dist(start, end) < float(minimum_length):
                    continue
                candidates.append(
                    ReadoutWall(
                        start=(float(start[0]), float(start[1])),
                        end=(float(end[0]), float(end[1])),
                        thickness=float(thickness),
                        wall_type="interior",
                    )
                )
        return tuple(candidates)

    def coverage(self, minimum_length: float = 2.5) -> dict[str, float | int]:
        accepted = [
            edge for edge in self.boundaries if edge.length >= minimum_length
        ]
        interior = [edge for edge in accepted if edge.edge_type == "interior"]
        hosted = sum(edge.owner_b >= 0 for edge in interior)
        return {
            "boundary_count": len(accepted),
            "interior_boundary_count": len(interior),
            "owner_pair_boundary_rate": (
                float(hosted / len(interior)) if interior else 1.0
            ),
            "junction_count": len(self.junctions),
            "adjacency_count": len(self.adjacency),
        }


@dataclass(frozen=True)
class _RawBoundary:
    start: tuple[float, float]
    end: tuple[float, float]
    owner_a: int
    owner_b: int
    edge_type: str
    confidence: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


def _owner_or_void(
    owner: np.ndarray,
    support: np.ndarray,
    a_rows: np.ndarray | slice,
    a_columns: np.ndarray | slice,
    b_rows: np.ndarray | slice,
    b_columns: np.ndarray | slice,
) -> tuple[np.ndarray, np.ndarray]:
    support_a = support[a_rows, a_columns]
    support_b = support[b_rows, b_columns]
    owner_a = np.where(support_a, owner[a_rows, a_columns], -1)
    owner_b = np.where(support_b, owner[b_rows, b_columns], -1)
    return owner_a.astype(np.int64), owner_b.astype(np.int64)


def _boundary_runs(
    owner: np.ndarray,
    support: np.ndarray,
    owner_confidence: np.ndarray | None,
    coordinate_scale: float,
) -> list[_RawBoundary]:
    raw: list[_RawBoundary] = []
    height, width = owner.shape
    confidence = (
        np.ones_like(owner, dtype=np.float32)
        if owner_confidence is None
        else np.asarray(owner_confidence, dtype=np.float32)
    )
    for fixed in range(1, height):
        owner_a, owner_b = _owner_or_void(
            owner,
            support,
            fixed - 1,
            np.s_[:],
            fixed,
            np.s_[:],
        )
        valid = (owner_a >= 0) | (owner_b >= 0)
        changed = np.flatnonzero(valid & (owner_a != owner_b))
        for index in changed:
            a = int(owner_a[index])
            b = int(owner_b[index])
            raw.append(
                _RawBoundary(
                    start=(
                        float(index) * coordinate_scale,
                        float(fixed) * coordinate_scale,
                    ),
                    end=(
                        float(index + 1) * coordinate_scale,
                        float(fixed) * coordinate_scale,
                    ),
                    owner_a=a,
                    owner_b=b,
                    edge_type=(
                        "interior" if min(a, b) >= 0 else "void"
                    ),
                    confidence=float(
                        0.5
                        * (
                            confidence[fixed - 1, index]
                            + confidence[fixed, index]
                        )
                    ),
                )
            )
    for fixed in range(1, width):
        owner_a, owner_b = _owner_or_void(
            owner,
            support,
            np.s_[:],
            fixed - 1,
            np.s_[:],
            fixed,
        )
        valid = (owner_a >= 0) | (owner_b >= 0)
        changed = np.flatnonzero(valid & (owner_a != owner_b))
        for index in changed:
            a = int(owner_a[index])
            b = int(owner_b[index])
            raw.append(
                _RawBoundary(
                    start=(
                        float(fixed) * coordinate_scale,
                        float(index) * coordinate_scale,
                    ),
                    end=(
                        float(fixed) * coordinate_scale,
                        float(index + 1) * coordinate_scale,
                    ),
                    owner_a=a,
                    owner_b=b,
                    edge_type="interior" if min(a, b) >= 0 else "void",
                    confidence=float(
                        0.5
                        * (
                            confidence[index, fixed - 1]
                            + confidence[index, fixed]
                        )
                    ),
                )
            )
    for fixed in (0, height):
        row = 0 if fixed == 0 else height - 1
        for index in np.flatnonzero(support[row, :]):
            owner_index = int(owner[row, index])
            start = (
                float(index) * coordinate_scale,
                float(fixed) * coordinate_scale,
            )
            end = (
                float(index + 1) * coordinate_scale,
                float(fixed) * coordinate_scale,
            )
            raw.append(
                _RawBoundary(
                    start=start,
                    end=end,
                    owner_a=-1,
                    owner_b=owner_index,
                    edge_type="void",
                    confidence=float(confidence[row, index]),
                )
            )
    for fixed in (0, width):
        column = 0 if fixed == 0 else width - 1
        for index in np.flatnonzero(support[:, column]):
            owner_index = int(owner[index, column])
            start = (
                float(fixed) * coordinate_scale,
                float(index) * coordinate_scale,
            )
            end = (
                float(fixed) * coordinate_scale,
                float(index + 1) * coordinate_scale,
            )
            raw.append(
                _RawBoundary(
                    start=start,
                    end=end,
                    owner_a=-1,
                    owner_b=owner_index,
                    edge_type="void",
                    confidence=float(confidence[index, column]),
                )
            )
    return _merge_collinear(raw)


def _merge_collinear(raw: list[_RawBoundary]) -> list[_RawBoundary]:
    grouped: dict[
        tuple[str, float, tuple[int, int], str],
        list[_RawBoundary],
    ] = {}
    for edge in raw:
        horizontal = edge.start[1] == edge.end[1]
        fixed = edge.start[1] if horizontal else edge.start[0]
        key = (
            "h" if horizontal else "v",
            fixed,
            tuple(sorted((edge.owner_a, edge.owner_b))),
            edge.edge_type,
        )
        grouped.setdefault(key, []).append(edge)
    merged: list[_RawBoundary] = []
    for edges in grouped.values():
        edges.sort(key=lambda edge: (edge.start[0], edge.start[1]))
        current = edges[0]
        for edge in edges[1:]:
            if edge.start != current.end:
                merged.append(current)
                current = edge
                continue
            current = _RawBoundary(
                start=current.start,
                end=edge.end,
                owner_a=current.owner_a,
                owner_b=current.owner_b,
                edge_type=current.edge_type,
                confidence=0.5
                * (current.confidence + edge.confidence),
            )
        merged.append(current)
    return merged


def _split_at_endpoints(
    boundaries: list[_RawBoundary],
    tolerance: float,
) -> list[_RawBoundary]:
    vertices: list[tuple[float, float]] = []
    for edge in boundaries:
        for point in (edge.start, edge.end):
            if all(math.dist(point, item) > tolerance for item in vertices):
                vertices.append(point)
    split: list[_RawBoundary] = []
    for edge in boundaries:
        start = np.asarray(edge.start, dtype=np.float64)
        end = np.asarray(edge.end, dtype=np.float64)
        direction = end - start
        length_squared = float(direction @ direction)
        if length_squared <= 1.0e-12:
            continue
        fractions = [0.0, 1.0]
        for point in vertices:
            value = np.asarray(point, dtype=np.float64)
            fraction = float((value - start) @ direction / length_squared)
            if fraction <= 1.0e-9 or fraction >= 1.0 - 1.0e-9:
                continue
            projected = start + fraction * direction
            if np.linalg.norm(value - projected) <= tolerance:
                fractions.append(fraction)
        fractions = sorted(set(fractions))
        for low, high in pairwise(fractions):
            split.append(
                _RawBoundary(
                    start=tuple(start + low * direction),
                    end=tuple(start + high * direction),
                    owner_a=edge.owner_a,
                    owner_b=edge.owner_b,
                    edge_type=edge.edge_type,
                    confidence=edge.confidence,
                )
            )
    return split


def _junction_clusters(
    boundaries: list[SharedBoundaryEdge],
    tolerance: float,
) -> tuple[OwnerJunction, ...]:
    endpoints = []
    for edge in boundaries:
        endpoints.extend(
            (
                (edge.start, edge.id),
                (edge.end, edge.id),
            )
        )
    clusters: list[dict[str, object]] = []
    for point, edge_id in endpoints:
        for cluster in clusters:
            center = cluster["center"]
            assert isinstance(center, tuple)
            if math.dist(point, center) <= tolerance:
                cluster["points"].append(point)
                cluster["boundaries"].add(edge_id)
                cluster["center"] = (
                    float(np.mean([item[0] for item in cluster["points"]])),
                    float(np.mean([item[1] for item in cluster["points"]])),
                )
                break
        else:
            clusters.append(
                {
                    "points": [point],
                    "boundaries": {edge_id},
                    "center": point,
                }
            )
    junctions = []
    for index, cluster in enumerate(clusters, start=1):
        boundary_ids = tuple(sorted(cluster["boundaries"]))
        junctions.append(
            OwnerJunction(
                id=f"junction_{index:04d}",
                point=cluster["center"],
                degree=len(boundary_ids),
                boundary_ids=boundary_ids,
            )
        )
    return tuple(junctions)


def extract_owner_cell_complex(
    owner: np.ndarray,
    support: np.ndarray,
    room_id_by_owner: Mapping[int, str],
    *,
    coordinate_scale: float = 1.0,
    owner_confidence: np.ndarray | None = None,
    junction_tolerance: float = 1.0,
    minimum_boundary_length: float = 0.0,
) -> OwnerCellComplex:
    """Build one shared cell complex before independent polygon quantization."""

    owner_array = np.asarray(owner, dtype=np.int64)
    support_array = np.asarray(support, dtype=bool)
    if owner_array.ndim != 2 or owner_array.shape != support_array.shape:
        raise ValueError("owner and support must share one 2-D shape")
    mapping = {
        int(index): str(room_id)
        for index, room_id in room_id_by_owner.items()
    }
    if not mapping:
        raise ValueError("owner cell complex requires at least one room")
    if owner_confidence is not None:
        confidence_array = np.asarray(owner_confidence, dtype=np.float32)
        if confidence_array.shape != owner_array.shape:
            raise ValueError("owner_confidence and owner must share one shape")
    else:
        confidence_array = None
    accepted_owner = np.zeros_like(support_array, dtype=bool)
    for index in mapping:
        accepted_owner |= support_array & (owner_array == index)
    raw = _boundary_runs(
        owner_array,
        accepted_owner,
        confidence_array,
        float(coordinate_scale),
    )
    raw = _split_at_endpoints(raw, float(junction_tolerance))
    boundaries = tuple(
        SharedBoundaryEdge(
            id=f"boundary_{index:04d}",
            start=edge.start,
            end=edge.end,
            owner_a=edge.owner_a,
            owner_b=edge.owner_b,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
        )
        for index, edge in enumerate(raw, start=1)
        if edge.length >= float(minimum_boundary_length)
    )
    adjacency = tuple(
        sorted(
            {
                tuple(sorted((mapping[edge.owner_a], mapping[edge.owner_b])))
                for edge in boundaries
                if edge.edge_type == "interior"
                and edge.owner_a in mapping
                and edge.owner_b in mapping
            }
        )
    )
    junctions = _junction_clusters(boundaries, float(junction_tolerance))
    return OwnerCellComplex(
        owner=owner_array,
        support=support_array,
        coordinate_scale=float(coordinate_scale),
        room_id_by_owner=tuple(sorted(mapping.items())),
        boundaries=boundaries,
        adjacency=adjacency,
        junctions=junctions,
    )


__all__ = [
    "OwnerCellComplex",
    "OwnerJunction",
    "SharedBoundaryEdge",
    "extract_owner_cell_complex",
]
