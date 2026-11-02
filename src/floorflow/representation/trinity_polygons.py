"""Lossless rectilinear room decomposition for Trinity block geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import RoomInstance, extract_room_instances
from floorflow.representation.trinity_latent import TrinityRoomBatch
from trinity.floorplan.parameterize import xywh_to_z


@dataclass(frozen=True)
class RoomPart:
    room_index: int
    class_id: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


def _row_runs(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    runs = []
    for row in mask:
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        runs.append(
            [
                (int(changes[index]), int(changes[index + 1]))
                for index in range(0, len(changes), 2)
            ]
        )
    return runs


def decompose_mask_to_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Exactly cover a binary mask with disjoint axis-aligned rectangles."""

    mask = np.asarray(mask, dtype=bool)
    runs = _row_runs(mask)
    active: dict[tuple[int, int], tuple[int, int]] = {}
    rectangles = []
    for y, row_runs in enumerate(runs + [[]]):
        current = set(row_runs)
        for span in tuple(active):
            if span not in current:
                start_y, _ = active.pop(span)
                rectangles.append((span[0], start_y, span[1], y))
        for span in current:
            active.setdefault(span, (y, span[1]))
    merged = []
    for rectangle in sorted(rectangles, key=lambda item: (item[1], item[3], item[0])):
        x0, y0, x1, y1 = rectangle
        for index, (mx0, my0, mx1, my1) in enumerate(merged):
            if y0 == my0 and y1 == my1 and x0 == mx1:
                merged[index] = (mx0, my0, x1, my1)
                break
        else:
            merged.append(rectangle)
    return merged


def room_parts_from_labels(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int = 32,
    min_area: int = 8,
) -> list[RoomPart]:
    instances = sorted(
        extract_room_instances(labels, taxonomy, min_area=min_area),
        key=lambda instance: instance.area,
        reverse=True,
    )[:max_rooms]
    parts = []
    for room_index, instance in enumerate(instances):
        for x0, y0, x1, y1 in decompose_mask_to_rectangles(instance.mask):
            parts.append(
                RoomPart(
                    room_index=room_index,
                    class_id=instance.class_id,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )
    return parts


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _boundary(part: RoomPart, bounds, tolerance: int = 2) -> int:
    fx0, fy0, fx1, fy1 = bounds
    code = 0
    code |= int(abs(part.x0 - fx0) <= tolerance)
    code |= int(abs(part.x1 - fx1) <= tolerance) << 1
    code |= int(abs(part.y0 - fy0) <= tolerance) << 2
    code |= int(abs(part.y1 - fy1) <= tolerance) << 3
    return code


def _part_adjacency(parts: list[RoomPart]) -> np.ndarray:
    count = len(parts)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for left in range(count):
        a = parts[left]
        for right in range(left + 1, count):
            b = parts[right]
            gap_x = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
            gap_y = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
            if gap_x == 0 and gap_y == 0:
                adjacency[left, right] = adjacency[right, left] = 1.0
    return adjacency


def _clusters(adjacency: np.ndarray) -> np.ndarray:
    cluster = np.zeros(len(adjacency), dtype=np.int64)
    current = 0
    for start in range(len(adjacency)):
        if cluster[start]:
            continue
        current += 1
        stack = [start]
        cluster[start] = current
        while stack:
            node = stack.pop()
            for neighbour in np.flatnonzero(adjacency[node] > 0):
                if not cluster[neighbour]:
                    cluster[neighbour] = current
                    stack.append(int(neighbour))
    return cluster


def room_part_batch_from_labels(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    footprint: torch.Tensor,
    *,
    max_rooms: int = 32,
    max_parts: int = 256,
    min_area: int = 8,
) -> TrinityRoomBatch:
    if labels.ndim == 2:
        labels = labels[None]
    if footprint.ndim == 2:
        footprint = footprint[None, None]
    elif footprint.ndim == 3:
        footprint = footprint[:, None]
    room_lookup = {
        class_id: index for index, class_id in enumerate(taxonomy.room_ids)
    }
    rows = []
    for index in range(labels.shape[0]):
        label_map = labels[index].detach().cpu().numpy()
        area_map = footprint[index, 0].detach().cpu().numpy().astype(bool)
        parts = room_parts_from_labels(
            label_map,
            taxonomy,
            max_rooms=max_rooms,
            min_area=min_area,
        )
        if len(parts) > max_parts:
            raise ValueError(
                f"room decomposition produced {len(parts)} parts > {max_parts}"
            )
        bounds = _bounds(area_map)
        xywh = np.zeros((max_parts, 4), dtype=np.float32)
        areas = np.zeros(max_parts, dtype=np.float32)
        classes = np.zeros(max_parts, dtype=np.int64)
        room_uid = np.zeros(max_parts, dtype=np.int64)
        codes = np.zeros(max_parts, dtype=np.int64)
        valid = np.zeros(max_parts, dtype=bool)
        for part_index, part in enumerate(parts):
            xywh[part_index] = (
                part.x0,
                part.y0,
                part.x1 - part.x0,
                part.y1 - part.y0,
            )
            areas[part_index] = part.area
            classes[part_index] = room_lookup[part.class_id]
            room_uid[part_index] = part.room_index
            codes[part_index] = _boundary(part, bounds)
            valid[part_index] = True
        adjacency = np.zeros((max_parts, max_parts), dtype=np.float32)
        adjacency[: len(parts), : len(parts)] = _part_adjacency(parts)
        clusters = _clusters(adjacency)
        scale = float(np.sqrt(max(1.0, areas.sum())))
        z = np.zeros((max_parts, 3), dtype=np.float32)
        z[valid] = xywh_to_z(xywh[valid], areas[valid], scale)
        rows.append((z, areas, scale, xywh, valid, classes, room_uid, clusters, codes, adjacency))
    device = labels.device
    return TrinityRoomBatch(
        z=torch.from_numpy(np.stack([row[0] for row in rows])).to(device),
        area_targets=torch.from_numpy(np.stack([row[1] for row in rows])).to(device),
        scale=torch.tensor([row[2] for row in rows], device=device),
        xywh=torch.from_numpy(np.stack([row[3] for row in rows])).to(device),
        token_mask=torch.from_numpy(np.stack([row[4] for row in rows])).to(device),
        room_classes=torch.from_numpy(np.stack([row[5] for row in rows])).to(device),
        cluster_id=torch.from_numpy(np.stack([row[7] for row in rows])).to(device),
        boundary_code=torch.from_numpy(np.stack([row[8] for row in rows])).to(device),
        adjacency=torch.from_numpy(np.stack([row[9] for row in rows])).to(device),
        room_uid=torch.from_numpy(np.stack([row[6] for row in rows])).to(device),
    )


__all__ = [
    "RoomPart",
    "decompose_mask_to_rectangles",
    "room_part_batch_from_labels",
    "room_parts_from_labels",
]
