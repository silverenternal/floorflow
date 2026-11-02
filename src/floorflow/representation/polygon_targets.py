"""Room polygon targets and deterministic polygon rasterization."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import RoomInstance, extract_room_instances


def _ordered_polygon(mask: np.ndarray, points: int) -> np.ndarray | None:
    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 3:
        return None
    signed_area = 0.5 * (
        np.sum(contour[:, 0] * np.roll(contour[:, 1], -1))
        - np.sum(contour[:, 1] * np.roll(contour[:, 0], -1))
    )
    if signed_area < 0:
        contour = contour[::-1]
    start = np.lexsort((contour[:, 0], contour[:, 1]))[0]
    contour = np.roll(contour, -start, axis=0)
    closed = np.concatenate([contour, contour[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 1.0e-6:
        return None
    sample_positions = np.linspace(
        0.0,
        cumulative[-1],
        int(points),
        endpoint=False,
    )
    sampled = np.stack(
        [
            np.interp(sample_positions, cumulative, closed[:, 0]),
            np.interp(sample_positions, cumulative, closed[:, 1]),
        ],
        axis=1,
    )
    return sampled


def polygon_instance_targets(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    max_instances: int = 24,
    points: int = 24,
    min_area: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return room-index classes, normalized loops, and validity masks."""

    batch_classes = []
    batch_polygons = []
    batch_valid = []
    height, width = labels.shape[-2:]
    room_lookup = {
        class_id: index
        for index, class_id in enumerate(taxonomy.room_ids)
    }
    for item in labels.detach().cpu().numpy():
        instances = sorted(
            extract_room_instances(item, taxonomy, min_area=min_area),
            key=lambda instance: instance.area,
            reverse=True,
        )[:max_instances]
        classes = torch.zeros(max_instances, dtype=torch.long)
        polygons = torch.zeros(max_instances, points, 2)
        valid = torch.zeros(max_instances, dtype=torch.bool)
        for index, instance in enumerate(instances):
            polygon = _ordered_polygon(instance.mask, points)
            if polygon is None:
                continue
            polygon[:, 0] = polygon[:, 0] / max(1.0, width - 1.0)
            polygon[:, 1] = polygon[:, 1] / max(1.0, height - 1.0)
            classes[index] = room_lookup.get(instance.class_id, 0)
            polygons[index] = torch.from_numpy(polygon).float()
            valid[index] = True
        batch_classes.append(classes)
        batch_polygons.append(polygons)
        batch_valid.append(valid)
    return (
        torch.stack(batch_classes).to(labels.device),
        torch.stack(batch_polygons).to(labels.device, dtype=torch.float32),
        torch.stack(batch_valid).to(labels.device),
    )


def rasterize_room_polygons(
    polygons: torch.Tensor,
    classes: torch.Tensor,
    valid: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    height: int = 256,
    width: int = 256,
) -> torch.Tensor:
    """Rasterize room-index polygons into a mutually exclusive semantic map."""

    batch_size = polygons.shape[0]
    output = np.zeros((batch_size, height, width), dtype=np.uint16)
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    polygons_np = polygons.detach().cpu().numpy()
    classes_np = classes.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    for batch_index in range(batch_size):
        candidates = []
        for query in range(polygons.shape[1]):
            if not valid_np[batch_index, query]:
                continue
            points = np.rint(
                polygons_np[batch_index, query]
                * np.asarray([width - 1, height - 1])
            ).astype(np.int32)
            area = abs(float(cv2.contourArea(points)))
            candidates.append((area, query, points))
        for _area, query, points in sorted(candidates, reverse=True):
            class_id = int(room_ids[int(classes_np[batch_index, query])])
            cv2.fillPoly(output[batch_index], [points], class_id)
    return torch.from_numpy(output.astype(np.int64)).to(
        device=polygons.device,
        dtype=torch.long,
    )


def collapse_invalid_polygon_points(
    polygons: torch.Tensor,
    point_valid: torch.Tensor,
) -> torch.Tensor:
    """Move valid vertices first and pad by repeating the final valid point."""

    if polygons.shape[:3] != point_valid.shape:
        raise ValueError("polygon and validity query shapes must match")
    valid = point_valid.bool()
    point_count = polygons.shape[2]
    valid_count = valid.sum(dim=-1)
    sorted_indices = (~valid).long().argsort(dim=-1, stable=True)
    identity = torch.arange(
        point_count,
        device=polygons.device,
    ).view(1, 1, -1).expand_as(sorted_indices)
    indices = torch.where(
        valid_count.ge(3).unsqueeze(-1),
        sorted_indices,
        identity,
    )
    gathered = polygons.gather(
        2,
        indices[..., None].expand(-1, -1, -1, polygons.shape[-1]),
    )
    valid_position = identity < valid_count.unsqueeze(-1)
    last_valid_index = sorted_indices.gather(
        2,
        (valid_count - 1).clamp(0, point_count - 1).unsqueeze(-1),
    ).squeeze(-1)
    last_valid = polygons.gather(
        2,
        last_valid_index[..., None, None].expand(
            -1,
            -1,
            1,
            polygons.shape[-1],
        ),
    ).expand_as(polygons)
    collapsed = torch.where(valid_position.unsqueeze(-1), gathered, last_valid)
    return torch.where(
        valid_count.ge(3).unsqueeze(-1).unsqueeze(-1),
        collapsed,
        polygons,
    )


def decode_polygon_loops(
    predictions: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    height: int = 256,
    width: int = 256,
    validity_threshold: float = 0.0,
) -> torch.Tensor:
    """Decode query polygons by a global validity threshold and class argmax."""

    classes = predictions["class_logits"].argmax(dim=-1)
    valid = predictions["valid_logits"] > float(validity_threshold)
    return rasterize_room_polygons(
        predictions["polygons"],
        classes,
        valid,
        taxonomy,
        height=height,
        width=width,
    )


def instance_polygon(instance: RoomInstance, points: int = 24) -> np.ndarray | None:
    """Public helper used by structural audits."""

    return _ordered_polygon(instance.mask, points)
