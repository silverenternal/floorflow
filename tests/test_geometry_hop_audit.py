from collections import Counter

import numpy as np

from floorflow.data.geometry_cache import (
    FloorplanGeometry,
    PackedPolygons,
    audit_geometry_hop,
    pack_polygons,
    rasterize_room_polygons,
)
from floorflow.data.taxonomy import build_taxonomy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _empty() -> PackedPolygons:
    return PackedPolygons(
        points=np.zeros((0, 2), dtype=np.float32),
        offsets=np.zeros(1, dtype=np.int64),
    )


def test_geometry_hop_audit_is_consistent_for_matching_support() -> None:
    taxonomy = _taxonomy()
    room_id = taxonomy.room_ids[0]
    polygon = np.array(
        [[8.0, 8.0], [24.0, 8.0], [24.0, 24.0], [8.0, 24.0]],
        dtype=np.float32,
    )
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = room_id
    rooms = pack_polygons([polygon])
    support = rasterize_room_polygons(
        rooms,
        np.array([room_id], dtype=np.int64),
        32,
    ) > 0
    empty = _empty()
    geometry = FloorplanGeometry(
        target=target,
        footprint=support.astype(np.uint8),
        room_support=support.astype(np.uint8),
        occlusion_mask=np.zeros_like(target),
        rooms=rooms,
        room_class_ids=np.array([room_id], dtype=np.int64),
        room_source_names=np.array(["Bedroom"]),
        walls=empty,
        wall_centerlines=np.zeros((0, 4), dtype=np.float32),
        wall_thickness=np.zeros((0,), dtype=np.float32),
        wall_types=np.zeros((0,), dtype=np.int64),
        openings=empty,
        opening_centers=np.zeros((0, 2), dtype=np.float32),
        opening_widths=np.zeros((0,), dtype=np.float32),
        opening_class_ids=np.zeros((0,), dtype=np.int64),
        opening_source_names=np.zeros((0,), dtype=object),
        opening_wall_indices=np.zeros((0,), dtype=np.int64),
        opening_host_distances=np.zeros((0,), dtype=np.float32),
        furniture=empty,
        furniture_class_ids=np.zeros((0,), dtype=np.int64),
        furniture_source_names=np.zeros((0,), dtype=object),
    )
    report = audit_geometry_hop(geometry, taxonomy)
    assert report["consistent"] is True
    assert report["checks"]["taxonomy_consistent"] is True
    assert report["checks"]["room_pixels_outside_footprint"] == 0
