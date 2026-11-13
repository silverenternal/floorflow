"""Lightweight raster-cell polygonization shared by CPU geometry consumers."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def component_cell_polygon(mask: np.ndarray) -> Polygon:
    """Return the exact union of labeled pixel cells for one 4-connected mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return Polygon()
    rectangles = []
    for row, values in enumerate(mask):
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        rectangles.extend(
            box(float(start), float(row), float(end), float(row + 1))
            for start, end in zip(starts, ends, strict=True)
        )
    polygon = unary_union(rectangles)
    if polygon.geom_type != "Polygon":
        raise ValueError("a 4-connected cell mask must produce one polygon")
    return polygon
