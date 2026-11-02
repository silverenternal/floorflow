"""Vector furniture instances from query or raster CAD layers."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.ndimage import label
from shapely.geometry import MultiPolygon, Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import ReadoutFurniture

_FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)


def _room_labels_numpy(room_labels: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(room_labels):
        room_labels = room_labels.detach().cpu().numpy()
    return np.asarray(room_labels, dtype=np.int64)


def _component_polygon(
    mask: np.ndarray,
    *,
    simplify_px: float,
) -> tuple[tuple[float, float], ...] | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 3.0:
        return None
    if simplify_px > 0.0:
        contour = cv2.approxPolyDP(
            contour,
            float(simplify_px),
            True,
        )
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None
    return tuple(
        (float(point[0]), float(point[1]))
        for point in points
    )


def _instances_from_masks(
    masks: list[tuple[np.ndarray, int, float]],
    room_labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int,
    simplify_px: float,
    room_indices: list[int] | None = None,
) -> tuple[ReadoutFurniture, ...]:
    instances = []
    for mask_index, (component_mask, class_index, confidence) in enumerate(masks):
        assigned_room = (
            None
            if room_indices is None
            else int(room_indices[mask_index])
        )
        room_rows = (
            [(assigned_room, taxonomy.room_ids[assigned_room])]
            if assigned_room is not None
            and 0 <= assigned_room < len(taxonomy.room_ids)
            else list(enumerate(taxonomy.room_ids))
        )
        for room_index, room_id in room_rows:
            room_mask = component_mask & (room_labels == room_id)
            if not room_mask.any():
                continue
            components, count = label(
                room_mask,
                structure=_FOUR_CONNECTED,
            )
            for component_id in range(1, int(count) + 1):
                mask = components == component_id
                area = int(mask.sum())
                if area < int(min_area):
                    continue
                polygon = _component_polygon(mask, simplify_px=simplify_px)
                if polygon is None:
                    continue
                instances.append(
                    (
                        area,
                        ReadoutFurniture(
                            label=taxonomy.class_names[
                                taxonomy.furniture_ids[class_index]
                            ],
                            polygon=polygon,
                            room_index=room_index,
                            confidence=float(confidence),
                        ),
                    )
                )
    instances.sort(key=lambda item: item[0], reverse=True)
    return tuple(item for _area, item in instances)


def read_furniture_queries(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    room_labels: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    query_budget: torch.Tensor | int | None = None,
    active_queries: torch.Tensor | None = None,
    room_assignment_logits: torch.Tensor | None = None,
    presence_threshold: float = 0.2,
    mask_threshold: float = 0.5,
    min_area: int = 8,
    simplify_px: float = 1.0,
) -> tuple[ReadoutFurniture, ...]:
    """Convert one sample's instance queries into room-attached polygons."""

    if class_logits.ndim != 2 or mask_logits.ndim != 3:
        raise ValueError("expected class_logits [Q,K+1] and mask_logits [Q,H,W]")
    if class_logits.shape[0] != mask_logits.shape[0]:
        raise ValueError("furniture query dimensions disagree")
    if class_logits.shape[-1] != len(taxonomy.furniture_ids) + 1:
        raise ValueError("furniture class logits do not match taxonomy")
    labels = _room_labels_numpy(room_labels)
    if labels.shape != tuple(mask_logits.shape[-2:]):
        raise ValueError("room labels and furniture masks must share one frame")
    if len(taxonomy.furniture_ids) == 0:
        return ()
    active_count = class_logits.shape[0]
    if query_budget is not None:
        if torch.is_tensor(query_budget):
            active_count = int(query_budget.detach().cpu().item())
        else:
            active_count = int(query_budget)
    active_count = max(0, min(active_count, class_logits.shape[0]))
    probability = class_logits.softmax(dim=-1)[..., :-1]
    confidence, classes = probability.max(dim=-1)
    masks = mask_logits.sigmoid()
    query_score = confidence * masks.flatten(1).amax(dim=-1)
    if room_assignment_logits is not None:
        assignment_probability = room_assignment_logits.softmax(dim=-1)
        query_score = query_score * (
            1.0 - assignment_probability[..., -1]
        )
    query_order = query_score.argsort(descending=True, stable=True).tolist()
    if active_queries is not None:
        active = active_queries.detach().bool().reshape(-1)
        if active.shape != (class_logits.shape[0],):
            raise ValueError("active_queries must match the query count")
        if not bool(active.any()):
            return ()
        if query_budget is not None:
            budget = (
                int(query_budget.detach().cpu().item())
                if torch.is_tensor(query_budget)
                else int(query_budget)
            )
            if budget != int(active.sum()):
                raise ValueError(
                    "query_budget and active_queries disagree"
                )
        query_order = [
            query
            for query in query_order
            if bool(active[query])
        ]
        active_count = len(query_order)
    rows = []
    room_indices = []
    for query in query_order[:active_count]:
        query_confidence = float(confidence[query].detach().cpu())
        if query_confidence < float(presence_threshold):
            continue
        mask = masks[query].detach().cpu().numpy() >= float(mask_threshold)
        if not mask.any():
            continue
        room_index = None
        if room_assignment_logits is not None:
            room_index = int(
                room_assignment_logits[query].argmax().detach().cpu()
            )
            if room_index >= len(taxonomy.room_ids):
                room_index = None
        rows.append(
            (
                mask,
                int(classes[query].detach().cpu()),
                query_confidence,
            )
        )
        room_indices.append(
            -1 if room_index is None else room_index
        )
    return _instances_from_masks(
        rows,
        labels,
        taxonomy,
        min_area=min_area,
        simplify_px=simplify_px,
        room_indices=(
            None
            if room_assignment_logits is None
            else room_indices
        ),
    )


def read_furniture_raster(
    furniture_logits: torch.Tensor,
    room_labels: torch.Tensor | np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.5,
    min_area: int = 8,
    simplify_px: float = 1.0,
) -> tuple[ReadoutFurniture, ...]:
    """Fallback vectorization for a dense furniture layer."""

    if furniture_logits.ndim != 3:
        raise ValueError("expected furniture_logits [C,H,W]")
    if furniture_logits.shape[0] != len(taxonomy.furniture_ids):
        raise ValueError("furniture logits do not match taxonomy")
    labels = _room_labels_numpy(room_labels)
    if labels.shape != tuple(furniture_logits.shape[-2:]):
        raise ValueError("room labels and furniture logits must share one frame")
    probability = furniture_logits.softmax(dim=0)
    confidence, classes = probability.max(dim=0)
    rows = []
    for class_index in range(len(taxonomy.furniture_ids)):
        mask = (
            (classes == class_index)
            & (confidence >= float(threshold))
        ).detach().cpu().numpy()
        if mask.any():
            rows.append((mask, class_index, float(threshold)))
    return _instances_from_masks(
        rows,
        labels,
        taxonomy,
        min_area=min_area,
        simplify_px=simplify_px,
    )


def clip_furniture_to_rooms(
    furniture: tuple[ReadoutFurniture, ...],
    rooms: list[dict] | tuple[dict, ...],
    *,
    taxonomy: FloorplanTaxonomy | None = None,
    minimum_area: float = 4.0,
) -> tuple[ReadoutFurniture, ...]:
    """Clip vector furniture to its assigned room as a CAD legality step."""

    clipped = []
    for item in furniture:
        candidates = list(range(len(rooms)))
        if taxonomy is not None and 0 <= item.room_index < len(taxonomy.room_ids):
            expected_label = taxonomy.class_names[
                taxonomy.room_ids[item.room_index]
            ]
            matching = [
                index
                for index, room in enumerate(rooms)
                if str(room["label"]) == expected_label
            ]
            if matching:
                candidates = matching
        if not candidates:
            continue
        polygon = Polygon(item.polygon)
        scored = []
        for room_index in candidates:
            room_polygon = Polygon(rooms[room_index]["polygon"])
            overlap = (
                polygon.intersection(room_polygon).area
                if polygon.is_valid and room_polygon.is_valid
                else 0.0
            )
            scored.append((overlap, room_index, room_polygon))
        _overlap, room_index, room_polygon = max(
            scored,
            key=lambda value: (value[0], -value[1]),
        )
        if not polygon.is_valid or not room_polygon.is_valid:
            continue
        intersection = polygon.intersection(room_polygon)
        if isinstance(intersection, MultiPolygon):
            intersection = max(intersection.geoms, key=lambda part: part.area)
        if not isinstance(intersection, Polygon) or intersection.area < minimum_area:
            continue
        clipped.append(
            ReadoutFurniture(
                label=item.label,
                polygon=tuple(
                    (float(point[0]), float(point[1]))
                    for point in intersection.exterior.coords[:-1]
                ),
                room_index=room_index,
                confidence=item.confidence,
            )
        )
    return tuple(clipped)


__all__ = [
    "clip_furniture_to_rooms",
    "read_furniture_queries",
    "read_furniture_raster",
]
