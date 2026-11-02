"""Channel-native zero-tolerance projection for edited floorplans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.vector_edit_protocol import VectorPlan
from floorflow.inference.local_delta import compose_local_delta
from floorflow.inference.vector_topology_projection import (
    project_edit_topology,
    repair_edited_room_polygons,
)


@dataclass(frozen=True)
class StrictLocalityProjection:
    controls: dict[str, torch.Tensor]
    labels: np.ndarray
    vector_plan: VectorPlan
    outside_changes: dict[str, int]


def native_region(
    region: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize a binary region without interpolation or boundary expansion."""

    source_height, source_width = source_size
    target_height, target_width = target_size
    source = np.asarray(region, dtype=bool)
    if source.shape != source_size:
        raise ValueError("source region shape mismatch")
    rows, columns = np.nonzero(source)
    output = np.zeros(target_size, dtype=bool)
    if rows.size == 0:
        return output
    top = int(np.floor(rows.min() * target_height / source_height))
    bottom = int(
        np.ceil((rows.max() + 1) * target_height / source_height)
    )
    left = int(np.floor(columns.min() * target_width / source_width))
    right = int(
        np.ceil((columns.max() + 1) * target_width / source_width)
    )
    output[
        max(top, 0) : min(bottom, target_height),
        max(left, 0) : min(right, target_width),
    ] = True
    return output


def native_control_regions(
    before: Mapping[str, torch.Tensor],
    region: np.ndarray,
) -> dict[str, np.ndarray]:
    region = np.asarray(region, dtype=bool)
    masks = {
        "program": region.copy(),
    }
    for key in ("room_logits", "structure", "furniture", "support"):
        masks[key] = native_region(
            region,
            tuple(region.shape),
            tuple(before[key].shape[-2:]),
        )
    return masks


def project_controls(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    regions: Mapping[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    output = {}
    for key in ("room_logits", "structure", "furniture", "support"):
        region = torch.from_numpy(np.asarray(regions[key], dtype=bool))
        output[key] = compose_local_delta(before[key], after[key], region)
    output["program"] = before["program"].clone()
    return output


def project_labels(
    before_labels: np.ndarray,
    after_labels: np.ndarray,
    room_labels: np.ndarray,
    region: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    before = np.asarray(before_labels, dtype=np.int64)
    after = np.asarray(after_labels, dtype=np.int64)
    output = after.copy()
    output[~np.asarray(region, dtype=bool)] = before[
        ~np.asarray(region, dtype=bool)
    ]
    projected = project_edit_topology(
        before,
        output,
        np.asarray(room_labels, dtype=np.int64),
        np.asarray(region, dtype=bool),
        np.asarray(footprint, dtype=bool),
        taxonomy,
        fill_holes=True,
        project_walls=True,
    ).labels
    projected[~np.asarray(region, dtype=bool)] = before[
        ~np.asarray(region, dtype=bool)
    ]
    return projected


def project_vector_plan(
    before: VectorPlan,
    after: VectorPlan,
    region: np.ndarray,
    footprint: np.ndarray,
) -> VectorPlan:
    region = np.asarray(region, dtype=bool)
    rooms = (
        tuple(
            room
            for room in before.rooms
            if not bool((room.mask & region).any())
        )
        + tuple(
            room
            for room in after.rooms
            if bool((room.mask & region).any())
        )
    )
    repaired = repair_edited_room_polygons(
        VectorPlan(
            rooms=rooms,
            walls=after.walls,
            openings=after.openings,
            wall_mask=after.wall_mask,
            opening_mask=after.opening_mask,
            source=after.source,
        ),
        region,
        np.asarray(footprint, dtype=bool),
    )["plan"]
    walls = _merge_region_vector_items(
        before.walls,
        repaired.walls,
        lambda wall: _line_touches_region(wall, region),
    )
    openings = _merge_region_vector_items(
        before.openings,
        repaired.openings,
        lambda opening: opening_interval_touches_region(
            opening,
            region,
            before.walls,
        ),
        touches_after=(
            lambda opening: opening_interval_touches_region(
                opening,
                region,
                repaired.walls,
            )
        ),
    )
    return VectorPlan(
        rooms=repaired.rooms,
        walls=walls,
        openings=openings,
        wall_mask=np.where(region, repaired.wall_mask, before.wall_mask),
        opening_mask=np.where(
            region,
            repaired.opening_mask,
            before.opening_mask,
        ),
        source=repaired.source,
    )


def _merge_region_vector_items(
    before,
    after,
    touches,
    *,
    touches_after=None,
):
    preserved = tuple(item for item in before if not touches(item))
    edited = tuple(
        item
        for item in after
        if (
            touches if touches_after is None else touches_after
        )(item)
    )
    return preserved + edited


def _line_touches_region(wall, region: np.ndarray) -> bool:
    start = np.asarray(wall.start, dtype=np.float64)
    end = np.asarray(wall.end, dtype=np.float64)
    samples = max(2, int(np.ceil(float(np.linalg.norm(end - start)))) + 1)
    for fraction in np.linspace(0.0, 1.0, samples):
        point = start + float(fraction) * (end - start)
        column = min(max(round(float(point[0])), 0), region.shape[1] - 1)
        row = min(max(round(float(point[1])), 0), region.shape[0] - 1)
        if bool(region[row, column]):
            return True
    return False


def opening_interval_touches_region(
    opening,
    region: np.ndarray,
    walls,
) -> bool:
    """Check the full opening interval using its nearest host-wall direction."""

    region = np.asarray(region, dtype=bool)
    center = np.asarray(opening.center, dtype=np.float64)

    def touches(point: np.ndarray) -> bool:
        column = min(max(round(float(point[0])), 0), region.shape[1] - 1)
        row = min(max(round(float(point[1])), 0), region.shape[0] - 1)
        return bool(region[row, column])

    if touches(center):
        return True
    host = None
    host_distance = float("inf")
    for wall in walls:
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        length_sq = float(delta @ delta)
        if length_sq <= 1.0e-9:
            continue
        fraction = float(
            np.clip((center - start) @ delta / length_sq, 0.0, 1.0)
        )
        distance = float(np.linalg.norm(center - (start + fraction * delta)))
        if distance < host_distance:
            host = (start, end)
            host_distance = distance
    if host is None:
        return False
    start, end = host
    delta = end - start
    direction = delta / max(float(np.linalg.norm(delta)), 1.0e-9)
    half_width = max(0.0, float(opening.width)) * 0.5
    samples = max(2, int(np.ceil(float(opening.width))) + 1)
    return any(
        touches(center + float(offset) * direction)
        for offset in np.linspace(-half_width, half_width, samples)
    )


def typed_edge_exact(
    before_adjacency: torch.Tensor,
    after_adjacency: torch.Tensor,
    unaffected: torch.Tensor,
) -> bool:
    mask = unaffected.bool()
    if mask.ndim == 1:
        mask = mask[:, None] & mask[None, :]
    return bool(
        torch.equal(
            before_adjacency.bool()[mask],
            after_adjacency.bool()[mask],
        )
    )


def outside_change_counts(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    regions: Mapping[str, np.ndarray],
) -> dict[str, int]:
    counts = {}
    for key in ("room_logits", "structure", "furniture", "support"):
        difference = (after[key] != before[key])
        while difference.ndim > 2:
            difference = difference.any(dim=0)
        outside = ~torch.from_numpy(
            np.asarray(regions[key], dtype=bool)
        ).to(difference.device)
        counts[key] = int((difference & outside).sum())
    return counts


__all__ = [
    "StrictLocalityProjection",
    "native_control_regions",
    "native_region",
    "opening_interval_touches_region",
    "outside_change_counts",
    "project_controls",
    "project_labels",
    "project_vector_plan",
    "typed_edge_exact",
]
