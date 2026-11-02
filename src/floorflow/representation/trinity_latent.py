"""CubiCasa room rasters to Trinity's area-preserving layout latent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import RoomInstance, extract_room_instances
from trinity.floorplan.parameterize import xywh_to_z, z_to_xywh


@dataclass(frozen=True)
class TrinityRoomBatch:
    z: torch.Tensor
    area_targets: torch.Tensor
    scale: torch.Tensor
    xywh: torch.Tensor
    token_mask: torch.Tensor
    room_classes: torch.Tensor
    cluster_id: torch.Tensor
    boundary_code: torch.Tensor
    adjacency: torch.Tensor
    room_uid: torch.Tensor | None = None

    @property
    def constraints(self) -> torch.Tensor:
        batch, count = self.token_mask.shape
        zeros = torch.zeros(
            batch,
            count,
            5,
            device=self.token_mask.device,
            dtype=self.token_mask.dtype,
        )
        zeros[..., 3] = self.cluster_id.float() * self.token_mask.float()
        zeros[..., 4] = self.boundary_code.float()
        return zeros


def _boundary_code(
    bbox: tuple[int, int, int, int],
    footprint_bounds: tuple[int, int, int, int],
    tolerance: int,
) -> int:
    x0, y0, x1, y1 = bbox
    fx0, fy0, fx1, fy1 = footprint_bounds
    code = 0
    if abs(x0 - fx0) <= tolerance:
        code |= 1
    if abs(x1 - fx1) <= tolerance:
        code |= 2
    if abs(y0 - fy0) <= tolerance:
        code |= 4
    if abs(y1 - fy1) <= tolerance:
        code |= 8
    return code


def _footprint_bounds(footprint: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(footprint.astype(bool))
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _adjacency(instances: list[RoomInstance]) -> np.ndarray:
    count = len(instances)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for left in range(count):
        near_left = binary_dilation(instances[left].mask, iterations=1)
        for right in range(left + 1, count):
            contact = np.logical_and(
                near_left,
                binary_dilation(instances[right].mask, iterations=1),
            ).sum()
            if contact:
                weight = float(contact)
                adjacency[left, right] = weight
                adjacency[right, left] = weight
    return adjacency


def _clusters(adjacency: np.ndarray) -> np.ndarray:
    count = adjacency.shape[0]
    cluster = np.zeros(count, dtype=np.int64)
    current = 0
    for start in range(count):
        if cluster[start]:
            continue
        current += 1
        stack = [start]
        cluster[start] = current
        while stack:
            node = stack.pop()
            neighbours = np.flatnonzero(adjacency[node] > 0)
            for neighbour in neighbours:
                if not cluster[neighbour]:
                    cluster[neighbour] = current
                    stack.append(int(neighbour))
    return cluster


def room_instances_from_labels(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int = 32,
    min_area: int = 8,
) -> list[RoomInstance]:
    instances = sorted(
        extract_room_instances(labels, taxonomy, min_area=min_area),
        key=lambda instance: instance.area,
        reverse=True,
    )
    return instances[: int(max_rooms)]


def room_batch_from_labels(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    footprint: torch.Tensor | None = None,
    *,
    max_rooms: int = 32,
    min_area: int = 8,
    boundary_tolerance: int = 2,
    device: torch.device | str | None = None,
) -> TrinityRoomBatch:
    if labels.ndim == 2:
        labels = labels[None]
    if footprint is None:
        footprint = torch.isin(
            labels,
            torch.tensor(taxonomy.room_ids, device=labels.device),
        )
    if footprint.ndim == 2:
        footprint = footprint[None, None]
    elif footprint.ndim == 3:
        footprint = footprint[:, None]
    target_device = labels.device if device is None else torch.device(device)
    room_lookup = {
        class_id: index for index, class_id in enumerate(taxonomy.room_ids)
    }
    rows = []
    for sample_index in range(labels.shape[0]):
        label_map = labels[sample_index].detach().cpu().numpy()
        area_map = footprint[sample_index, 0].detach().cpu().numpy().astype(bool)
        instances = room_instances_from_labels(
            label_map,
            taxonomy,
            max_rooms=max_rooms,
            min_area=min_area,
        )
        bounds = _footprint_bounds(area_map)
        xywh = np.zeros((max_rooms, 4), dtype=np.float32)
        areas = np.zeros(max_rooms, dtype=np.float32)
        classes = np.zeros(max_rooms, dtype=np.int64)
        codes = np.zeros(max_rooms, dtype=np.int64)
        valid = np.zeros(max_rooms, dtype=bool)
        for index, instance in enumerate(instances):
            x0, y0, x1, y1 = instance.bbox
            xywh[index] = (x0, y0, x1 - x0, y1 - y0)
            areas[index] = float(instance.area)
            classes[index] = room_lookup[instance.class_id]
            codes[index] = _boundary_code(
                instance.bbox,
                bounds,
                boundary_tolerance,
            )
            valid[index] = True
        adjacency = np.zeros((max_rooms, max_rooms), dtype=np.float32)
        adjacency[: len(instances), : len(instances)] = _adjacency(instances)
        clusters = _clusters(adjacency)
        scale = float(np.sqrt(max(1.0, areas.sum())))
        z = np.zeros((max_rooms, 3), dtype=np.float32)
        z[valid] = xywh_to_z(xywh[valid], areas[valid], scale)
        rows.append(
            {
                "z": z,
                "area": areas,
                "scale": scale,
                "xywh": xywh,
                "valid": valid,
                "classes": classes,
                "clusters": clusters,
                "code": codes,
                "adjacency": adjacency,
            }
        )
    return TrinityRoomBatch(
        z=torch.from_numpy(np.stack([row["z"] for row in rows])).to(target_device),
        area_targets=torch.from_numpy(
            np.stack([row["area"] for row in rows])
        ).to(target_device),
        scale=torch.tensor(
            [row["scale"] for row in rows],
            device=target_device,
            dtype=torch.float32,
        ),
        xywh=torch.from_numpy(
            np.stack([row["xywh"] for row in rows])
        ).to(target_device),
        token_mask=torch.from_numpy(
            np.stack([row["valid"] for row in rows])
        ).to(target_device),
        room_classes=torch.from_numpy(
            np.stack([row["classes"] for row in rows])
        ).to(target_device),
        cluster_id=torch.from_numpy(
            np.stack([row["clusters"] for row in rows])
        ).to(target_device),
        boundary_code=torch.from_numpy(
            np.stack([row["code"] for row in rows])
        ).to(target_device),
        adjacency=torch.from_numpy(
            np.stack([row["adjacency"] for row in rows])
        ).to(target_device),
        room_uid=torch.arange(
            max_rooms,
            device=target_device,
        )[None].expand(len(rows), -1),
    )


def decode_room_boxes(
    z: np.ndarray,
    area_targets: np.ndarray,
    scale: float,
) -> np.ndarray:
    return z_to_xywh(z, area_targets, scale)


def render_room_boxes(
    boxes: torch.Tensor,
    room_classes: torch.Tensor,
    token_mask: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Paint Trinity room rectangles into the project's label raster."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=boxes.device,
        dtype=torch.long,
    )
    labels = torch.zeros(
        boxes.shape[0],
        *footprint.shape[-2:],
        device=boxes.device,
        dtype=torch.long,
    )
    areas = boxes[..., 2] * boxes[..., 3]
    order = areas.masked_fill(~token_mask, float("inf")).argsort(dim=1)
    height, width = labels.shape[-2:]
    for batch_index in range(boxes.shape[0]):
        for room_index in order[batch_index].tolist():
            if not bool(token_mask[batch_index, room_index]):
                continue
            x, y, box_width, box_height = boxes[batch_index, room_index]
            x0 = max(0, min(width, int(torch.floor(x).item())))
            y0 = max(0, min(height, int(torch.floor(y).item())))
            x1 = max(0, min(width, int(torch.ceil(x + box_width).item())))
            y1 = max(0, min(height, int(torch.ceil(y + box_height).item())))
            if x1 <= x0 or y1 <= y0:
                continue
            class_id = room_ids[room_classes[batch_index, room_index].long()]
            labels[batch_index, y0:y1, x0:x1] = class_id
    return labels * footprint[:, 0].long()


__all__ = [
    "TrinityRoomBatch",
    "decode_room_boxes",
    "render_room_boxes",
    "room_batch_from_labels",
    "room_instances_from_labels",
]
