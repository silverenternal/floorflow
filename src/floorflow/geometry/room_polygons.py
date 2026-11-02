"""Convert a room-label raster into explicit polygon room instances."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label
from shapely.geometry import Polygon

from floorflow.data.taxonomy import FloorplanTaxonomy


def inferred_footprint(labels: np.ndarray) -> np.ndarray:
    occupied = np.asarray(labels) > 0
    closed = binary_closing(occupied, structure=np.ones((5, 5), dtype=bool))
    filled = binary_fill_holes(closed)
    components, count = label(filled)
    if count <= 1:
        return filled
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(sizes.argmax())


def room_polygon_rows(
    labels: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    min_area: int = 12,
    epsilon: float = 1.5,
    footprint: np.ndarray | None = None,
    tile_footprint: bool = False,
) -> list[dict]:
    labels = np.asarray(labels, dtype=np.int64).copy()
    if tile_footprint and footprint is not None:
        area = np.asarray(footprint, dtype=bool)
        seeds = area & np.isin(labels, np.asarray(taxonomy.room_ids))
        if seeds.any():
            indices = distance_transform_edt(
                ~seeds,
                return_distances=False,
                return_indices=True,
            )
            filled = labels[tuple(indices)]
            labels[area] = filled[area]
    rows = []
    for class_id in taxonomy.room_ids:
        mask = (labels == class_id).astype(np.uint8)
        count, components = cv2.connectedComponents(mask, connectivity=4)
        for component_id in range(1, count):
            component = (components == component_id).astype(np.uint8)
            area = int(component.sum())
            if area < min_area:
                continue
            contours, _hierarchy = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            simplified = cv2.approxPolyDP(
                contour,
                epsilon=float(epsilon),
                closed=True,
            ).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            polygon = Polygon(simplified.astype(float))
            if not polygon.is_valid:
                repaired = polygon.buffer(0)
                if repaired.geom_type == "MultiPolygon":
                    repaired = max(repaired.geoms, key=lambda item: item.area)
                if repaired.is_empty or repaired.geom_type != "Polygon":
                    continue
                simplified = np.asarray(
                    repaired.exterior.coords[:-1],
                    dtype=np.float64,
                )
            rows.append(
                {
                    "label": taxonomy.class_names[class_id],
                    "walls": [],
                    "polygon": simplified.astype(float).tolist(),
                }
            )
    return rows


__all__ = ["inferred_footprint", "room_polygon_rows"]
