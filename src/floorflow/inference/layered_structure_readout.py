"""Compose explicit room polygons with an independently predicted wall layer."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.partition_polygons import partition_polygon_rows
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    raster_labels_to_wall_plan,
)


def build_layered_structure_plan(
    room_labels: np.ndarray,
    wall_labels: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> RasterWallPlan:
    """Read walls from a dedicated map and rooms from the unchanged room layer."""

    plan = raster_labels_to_wall_plan(wall_labels, taxonomy)
    rooms = partition_polygon_rows(room_labels, footprint, taxonomy)
    if rooms:
        plan.rooms = rooms
        plan.room_source = "layered_room_partition"
    return plan


__all__ = ["build_layered_structure_plan"]
