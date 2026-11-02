"""Adapt CE2EPlan RPLAN outputs to the shared FEVP evaluators."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from floorflow.data.rplan_boundary_adapter import RPLAN_CATEGORY_TO_ROOM
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutWall,
)

CE2E_CATEGORY_TO_ROOM = RPLAN_CATEGORY_TO_ROOM


def _category_id(category: int, taxonomy: FloorplanTaxonomy) -> int:
    name = CE2E_CATEGORY_TO_ROOM.get(int(category), "room:Undefined")
    return taxonomy.class_id(name)


def _boundary_xy(boundary: Sequence[Sequence[float]]) -> np.ndarray:
    return np.rint(
        np.asarray([[point[0], point[1]] for point in boundary], dtype=np.float32)
    ).astype(np.int32)


def _room_polygon(room: dict, *, prediction: bool) -> np.ndarray | None:
    boundary_key = "r_boundary_aligned" if prediction else "r_boundary"
    coordinates = room.get(boundary_key) or []
    if len(coordinates) >= 3:
        return np.rint(np.asarray(coordinates, dtype=np.float32)).astype(np.int32)
    box_key = "box_aligned" if prediction else "box"
    value = room.get(box_key) or []
    if len(value) != 4:
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.rint(
        np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    ).astype(np.int32)


def adapt_ce2eplan_row(
    row: dict,
    taxonomy: FloorplanTaxonomy,
    *,
    prediction: bool,
    image_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, RasterWallPlan, list[dict]]:
    """Rasterize one CE2EPlan row and build an explicit FEVP-compatible plan."""

    boundary = _boundary_xy(row["boundary"])
    footprint_image = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(footprint_image, [boundary], 1)
    labels = np.zeros((image_size, image_size), dtype=np.uint8)
    rooms = []
    order_key = "order_aligned" if prediction else "order"
    ordered = sorted(
        enumerate(row.get("rooms", [])),
        key=lambda item: (
            float(item[1].get(order_key, item[0])),
            item[0],
        ),
    )
    for _, room in ordered:
        polygon = _room_polygon(room, prediction=prediction)
        category = int(room.get("category", 4))
        class_id = _category_id(category, taxonomy)
        label = taxonomy.class_names[class_id]
        if polygon is not None:
            cv2.fillPoly(labels, [polygon], class_id)
            rooms.append(
                {
                    "label": label,
                    "polygon": polygon.astype(float).tolist(),
                    "walls": [],
                }
            )
    labels[footprint_image == 0] = 0
    wall_id = taxonomy.class_id("wall")
    cv2.polylines(
        labels,
        [boundary.reshape(-1, 1, 2)],
        isClosed=True,
        color=wall_id,
        thickness=2,
    )
    walls = [
        ReadoutWall(
            start=tuple(float(value) for value in boundary[index]),
            end=tuple(
                float(value)
                for value in boundary[(index + 1) % len(boundary)]
            ),
            thickness=2.0,
            wall_type="exterior",
        )
        for index in range(len(boundary))
    ]
    plan = RasterWallPlan(
        frame=float(image_size),
        walls=walls,
        rooms=rooms,
        openings=[],
        room_source="ce2eplan_explicit",
    )
    return labels, footprint_image, plan, rooms


__all__ = ["CE2E_CATEGORY_TO_ROOM", "adapt_ce2eplan_row"]
