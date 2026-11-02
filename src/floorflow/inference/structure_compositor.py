"""Deterministic readout policy for room, wall, junction, and opening layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as component_label

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.wall_junction import cluster_wall_endpoints
from floorflow.geometry.core import compose_known_exterior
from floorflow.geometry.owner_cell_complex import extract_owner_cell_complex
from floorflow.geometry.partition_polygons import (
    explicit_partition_plan,
    partition_polygon_rows,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    ReadoutWall,
    raster_labels_to_wall_plan,
)
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.room_repair import merge_room_fragments
from floorflow.representation.layered_floorplan import LayeredFloorplan

_FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class StructureReadoutPolicy:
    wall_threshold: float = 0.5
    wall_core_threshold: float = 0.0
    wall_tail_radius: int = 1
    opening_threshold: float = 0.5
    junction_threshold: float = 0.5
    junction_degree_min: int = 2
    seam_mode: str = "none"
    room_boundary_threshold: float = 0.5
    seam_radius: int = 0
    opening_support_radius: int = 0
    opening_min_area: int = 1
    exterior_mode: str = "fixed"
    exterior_width: int = 7
    reference_resolution: int = 384

    def validate(self) -> None:
        if self.seam_mode not in {
            "legacy",
            "none",
            "boundary_gated",
            "wall_gated",
        }:
            raise ValueError(f"unsupported seam mode: {self.seam_mode}")
        if self.exterior_mode not in {"fixed", "resolution_scaled"}:
            raise ValueError(
                f"unsupported exterior mode: {self.exterior_mode}"
            )
        for name in (
            "wall_threshold",
            "wall_core_threshold",
            "opening_threshold",
            "junction_threshold",
            "room_boundary_threshold",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.wall_core_threshold not in {0.0, self.wall_threshold} and (
            self.wall_core_threshold < self.wall_threshold
        ):
            raise ValueError(
                "wall_core_threshold must be zero or at least wall_threshold"
            )
        if min(
            self.seam_radius,
            self.wall_tail_radius,
            self.opening_support_radius,
            self.opening_min_area,
            self.exterior_width,
            self.reference_resolution,
        ) < 0:
            raise ValueError("readout sizes and radii must be non-negative")


def _binary_components(mask: np.ndarray, minimum_area: int):
    labels, count = component_label(mask, structure=_FOUR_CONNECTED)
    accepted = np.zeros_like(mask, dtype=bool)
    for index in range(1, int(count) + 1):
        component = labels == index
        if int(component.sum()) >= int(minimum_area):
            accepted |= component
    return accepted


def _edge_support(
    edge,
    evidence: np.ndarray,
    radius: int,
) -> float:
    if edge.edge_type == "void":
        return 1.0
    start = np.asarray(edge.start, dtype=np.float64)
    end = np.asarray(edge.end, dtype=np.float64)
    count = max(1, int(np.linalg.norm(end - start) * 2.0))
    fractions = np.linspace(0.0, 1.0, count + 1)
    points = start[None] + fractions[:, None] * (end - start)[None]
    rows = points[:, 1].round().astype(np.int64)
    columns = points[:, 0].round().astype(np.int64)
    valid = (
        (rows >= 0)
        & (rows < evidence.shape[0])
        & (columns >= 0)
        & (columns < evidence.shape[1])
    )
    if not valid.any():
        return 0.0
    near = binary_dilation(evidence, iterations=max(0, int(radius)))
    return float(near[rows[valid], columns[valid]].mean())


def _project_opening(
    opening: ReadoutOpening,
    walls: list[ReadoutWall],
) -> ReadoutOpening | None:
    if not walls:
        return None
    point = np.asarray(opening.center, dtype=np.float64)
    best = None
    for index, wall in enumerate(walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        denominator = float(delta @ delta)
        fraction = (
            0.0
            if denominator <= 1.0e-12
            else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
        )
        projected = start + fraction * delta
        distance = float(np.linalg.norm(point - projected))
        if best is None or distance < best[0]:
            best = (distance, index, projected, math.dist(start, end))
    if best is None:
        return None
    _distance, index, center, wall_length = best
    return ReadoutOpening(
        center=(float(center[0]), float(center[1])),
        width=max(1.0, min(float(opening.width), wall_length)),
        kind=opening.kind,
        wall_index=int(index),
    )


def _filter_opening_components(
    opening: torch.Tensor,
    wall_support: torch.Tensor,
    minimum_area: int,
) -> torch.Tensor:
    if int(minimum_area) <= 1:
        return opening
    rows = []
    for opening_row, support_row in zip(
        opening.cpu().numpy(),
        wall_support.cpu().numpy(),
        strict=True,
    ):
        active = (opening_row != 0) & support_row
        filtered = _binary_components(active, int(minimum_area))
        rows.append(np.where(filtered, opening_row, 0))
    return torch.from_numpy(np.stack(rows)).to(opening.device)


def decode_structure_layers(
    room_logits: torch.Tensor,
    structure: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    policy: StructureReadoutPolicy | None = None,
) -> tuple[LayeredFloorplan, dict[str, torch.Tensor]]:
    """Apply one explicit policy to structure probabilities."""

    policy = policy or StructureReadoutPolicy()
    policy.validate()
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    target_size = footprint.shape[-2:]
    room_ids = torch.tensor(taxonomy.room_ids, device=room_logits.device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=room_logits.device)
    rooms = F.interpolate(
        room_ids[room_logits.argmax(dim=1)][:, None].float(),
        size=target_size,
        mode="nearest",
    )[:, 0].long()
    rooms *= footprint[:, 0].long()
    wall_probability = F.interpolate(
        structure["wall_logits"].sigmoid(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    wall = wall_probability >= float(policy.wall_threshold)
    core_wall = torch.zeros_like(wall)
    if policy.wall_core_threshold > float(policy.wall_threshold):
        core = wall_probability >= float(policy.wall_core_threshold)
        radius = 2 * int(policy.wall_tail_radius) + 1
        core_wall = F.max_pool2d(
            core.float()[:, None],
            radius,
            stride=1,
            padding=int(policy.wall_tail_radius),
        )[:, 0] > 0.5
        wall &= core_wall
    junction_support = torch.zeros_like(wall)
    if "junction_logits" in structure:
        junction_probability = F.interpolate(
            structure["junction_logits"].sigmoid(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        junction_support = (
            junction_probability >= float(policy.junction_threshold)
        )
        if "junction_degree_logits" in structure:
            degree = F.interpolate(
                structure["junction_degree_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            junction_support &= degree >= int(policy.junction_degree_min)
        wall |= junction_support
    seam = partition_wall_mask(rooms, radius=int(policy.seam_radius))
    if policy.seam_mode == "boundary_gated":
        if "room_boundary_logits" not in structure:
            raise ValueError(
                "boundary_gated seam requires room_boundary_logits"
            )
        boundary_probability = F.interpolate(
            structure["room_boundary_logits"].sigmoid(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        seam &= boundary_probability >= float(
            policy.room_boundary_threshold
        )
    elif policy.seam_mode == "wall_gated":
        seam &= wall_probability >= float(
            policy.room_boundary_threshold
        )
    if policy.seam_mode != "none":
        wall |= seam
    wall &= footprint[:, 0].bool()
    wall_support = F.max_pool2d(
        wall.float()[:, None],
        kernel_size=2 * int(policy.opening_support_radius) + 1,
        stride=1,
        padding=int(policy.opening_support_radius),
    )[:, 0] > 0.5
    opening_type = F.interpolate(
        structure["opening_type_logits"],
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    opening = opening_ids[opening_type] * (
        F.interpolate(
            structure["opening_logits"].sigmoid(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0] >= float(policy.opening_threshold)
    )
    opening = torch.where(
        wall_support,
        opening,
        torch.zeros_like(opening),
    )
    opening = _filter_opening_components(
        opening,
        wall_support,
        int(policy.opening_min_area),
    )
    layers = LayeredFloorplan(
        room_labels=rooms,
        wall=wall.long(),
        opening=opening,
        furniture=torch.zeros_like(rooms),
        footprint=footprint[:, 0],
        taxonomy=taxonomy,
    )
    return layers, {
        "wall_probability": wall_probability.mean(),
        "wall_core_rate": core_wall.float().mean(),
        "junction_support_rate": junction_support.float().mean(),
        "seam_rate": seam.float().mean(),
        "opening_support_rate": wall_support.float().mean(),
    }


def effective_exterior_width(
    spatial_size: tuple[int, int],
    policy: StructureReadoutPolicy,
) -> int:
    if policy.exterior_mode == "fixed":
        return max(1, int(policy.exterior_width))
    scale = min(spatial_size) / max(1, int(policy.reference_resolution))
    return max(1, round(float(policy.exterior_width) * scale))


def compose_structure_readout(
    layers: LayeredFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    policy: StructureReadoutPolicy | None = None,
) -> torch.Tensor:
    policy = policy or StructureReadoutPolicy()
    width = effective_exterior_width(layers.room_labels.shape[-2:], policy)
    return compose_known_exterior(
        layers.visible_labels(),
        layers.footprint,
        boundary_width=width,
        opening_ids=taxonomy.opening_ids,
    )


def structure_opening_family_outputs(
    structure: dict[str, torch.Tensor],
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose structure openings in the host-interval reader's tensor form."""

    opening_probability = F.interpolate(
        structure["opening_logits"].sigmoid(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    family_probability = F.interpolate(
        structure["opening_type_logits"].softmax(dim=1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).clamp_min(1.0e-8)
    return opening_probability, family_probability.log()


def owner_cell_plan(
    labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    minimum_area: int = 32,
    wall_support_radius: int = 1,
    edge_support_threshold: float = 0.25,
):
    """Build rooms, walls, and host indices from one owner-cell topology."""

    labels = merge_room_fragments(
        np.asarray(labels, dtype=np.int64),
        taxonomy,
        min_area=int(minimum_area),
    )
    footprint_mask = np.asarray(footprint, dtype=bool)
    raster_plan = raster_labels_to_wall_plan(labels, taxonomy)
    fallback = explicit_partition_plan(
        labels,
        footprint_mask,
        taxonomy,
        minimum_area=int(minimum_area),
    )
    dense_indices = build_room_partition_target(
        labels,
        footprint_mask,
        taxonomy,
        fill_occluded=True,
    )
    owner = np.full_like(dense_indices, -1, dtype=np.int64)
    mapping: dict[int, str] = {}
    label_by_room_id: dict[str, str] = {}
    next_owner = 1
    for class_index, class_id in enumerate(taxonomy.room_ids):
        components, count = component_label(
            dense_indices == class_index,
            structure=_FOUR_CONNECTED,
        )
        for component in range(1, int(count) + 1):
            mask = components == component
            if int(mask.sum()) < int(minimum_area):
                continue
            owner[mask] = next_owner
            room_id = f"room_{next_owner:04d}"
            mapping[next_owner] = room_id
            label_by_room_id[room_id] = taxonomy.class_names[int(class_id)]
            next_owner += 1
    if not mapping:
        return fallback
    support = footprint_mask & (owner >= 0)
    cell = extract_owner_cell_complex(
        owner,
        support,
        mapping,
        junction_tolerance=1.0,
        minimum_boundary_length=1.0,
    )
    wall_evidence = (labels == taxonomy.class_id("wall")) | np.isin(
        labels,
        np.asarray(taxonomy.opening_ids),
    )
    selected_edges = [
        edge
        for edge in cell.boundaries
        if _edge_support(
            edge,
            wall_evidence,
            int(wall_support_radius),
        )
        >= float(edge_support_threshold)
    ]
    walls = [
        ReadoutWall(
            start=edge.start,
            end=edge.end,
            thickness=3.0,
            wall_type=(
                "exterior" if edge.edge_type == "void" else "interior"
            ),
            continuity=float(edge.confidence),
        )
        for edge in selected_edges
    ]
    if not walls:
        return fallback
    rows = []
    for room_id, polygon in cell.room_polygons().items():
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        rows.append(
            {
                "id": room_id,
                "label": label_by_room_id[room_id],
                "polygon": np.asarray(
                    polygon.exterior.coords[:-1],
                    dtype=np.float64,
                ).tolist(),
                "holes": [
                    np.asarray(interior.coords[:-1]).tolist()
                    for interior in polygon.interiors
                ],
            }
        )
    if not rows:
        return fallback
    plan = RasterWallPlan(
        frame=float(max(labels.shape)),
        walls=walls,
        rooms=rows or partition_polygon_rows(
            labels,
            footprint_mask,
            taxonomy,
            minimum_area=int(minimum_area),
        ),
        openings=[],
        room_source="owner_cell_complex",
    )
    plan.owner_cell_complex = cell
    clustered_junctions = cluster_wall_endpoints(
        walls,
        tolerance=1.0,
    )
    plan.junctions = [
        {
            "id": f"junction_{index:04d}",
            "point": junction.center,
            "degree": junction.degree,
        }
        for index, junction in enumerate(clustered_junctions, start=1)
    ]
    plan.openings = [
        projected
        for opening in raster_plan.openings
        if (projected := _project_opening(opening, walls)) is not None
    ]
    if not plan.rooms:
        return fallback
    plan.room_source = "owner_cell_complex"
    return plan


__all__ = [
    "StructureReadoutPolicy",
    "compose_structure_readout",
    "decode_structure_layers",
    "effective_exterior_width",
    "owner_cell_plan",
    "structure_opening_family_outputs",
]
