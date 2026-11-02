"""Regression tests for fixed-point-safe mask partition geometry."""

import numpy as np
import torch
from shapely.geometry import GeometryCollection, LineString, Polygon, box

from floorflow.data.rplan_boundary_adapter import rplan_taxonomy
from floorflow.geometry.partition_polygons import component_cell_polygon
from floorflow.inference.mask_partition_readout import (
    _bounded_crack_repair,
    _bridge_support_components,
    _largest_polygon_geometry,
    _support_is_watertight,
    decode_mask_partition_rooms,
    fixed_point_safe_polygon,
)


def test_fixed_point_readout_repairs_quantized_footprint_clip() -> None:
    footprint = Polygon(
        [
            (22.360291, 17.369302),
            (11.767251, 23.778592),
            (7.824561, 16.649290),
            (4.727769, 18.117543),
            (6.219754, 13.981862),
            (18.272330, 11.421624),
            (23.798503, 11.030954),
        ]
    )
    component = np.zeros((24, 24), dtype=bool)
    component[9:14, 5:18] = True
    clipped = component_cell_polygon(component).intersection(
        footprint.intersection(box(5.0, 9.0, 18.0, 14.0))
    )
    rounded = Polygon(
        [(round(x * 16.0) / 16.0, round(y * 16.0) / 16.0) for x, y in clipped.exterior.coords[:-1]]
    )

    assert clipped.is_valid
    assert not rounded.is_valid
    repaired = fixed_point_safe_polygon(clipped)
    assert repaired.is_valid
    assert repaired.area > 14.0
    assert abs(repaired.area - clipped.area) < 0.5


def test_query_keeps_only_its_largest_room_component() -> None:
    logits = torch.full((1, 1, 16, 16), -8.0)
    logits[0, 0, 1:4, 1:4] = 8.0
    logits[0, 0, 10:12, 10:12] = 8.0
    boundary = np.asarray([[0, 0], [0, 16], [16, 16], [16, 0]], dtype=np.float32)
    rooms, operations, adjacency = decode_mask_partition_rooms(
        logits,
        boundary,
        [0],
        np.asarray([0]),
        torch.tensor([0.9]),
        rplan_taxonomy(),
        canvas_size=16,
        confidence_threshold=0.5,
        minimum_area=1,
        gap_policy="threshold",
    )

    assert len(rooms) == 1
    assert Polygon(rooms[0]["polygon"]).area > 7.8
    assert "room_drop_secondary_query_component" in {
        operation.kind for operation in operations
    }
    assert adjacency is None


def test_largest_polygon_geometry_accepts_mixed_clip_collection() -> None:
    mixed = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]).intersection(
        Polygon([(1, 1), (1, 4), (4, 4), (4, 1)])
    )
    tiny = Polygon([(8, 8), (8, 9), (9, 9), (9, 8)])
    collection = GeometryCollection([mixed, tiny, LineString([(20, 20), (21, 21)])])

    selected = _largest_polygon_geometry(collection)

    assert selected.geom_type == "Polygon"
    assert selected.area == 1.0


def test_bounded_crack_repair_bridges_cracks_without_exterior_dilation() -> None:
    support = np.zeros((9, 11), dtype=bool)
    support[2:7, 2:5] = True
    support[2:7, 6:9] = True

    repaired = _bounded_crack_repair(support)

    assert repaired[2:7, 5].all()
    assert repaired.sum() == support.sum() + 5
    assert not repaired[0].any()
    assert not repaired[-1].any()
    assert not repaired[:, 0].any()
    assert not repaired[:, -1].any()


def test_bounded_crack_repair_fills_only_small_bounded_holes() -> None:
    support = np.ones((9, 13), dtype=bool)
    support[3:5, 3:5] = False
    support[2:7, 7:12] = False

    repaired = _bounded_crack_repair(support)

    assert repaired[3:5, 3:5].all()
    assert not repaired[2:7, 7:12].any()


def test_bridge_fill_connects_substantial_support_islands_minimally() -> None:
    support = np.zeros((9, 13), dtype=bool)
    support[2:7, 2:5] = True
    support[2:7, 7:10] = True

    bridged = _bridge_support_components(support, minimum_component_area=1)

    assert _support_is_watertight(bridged)
    assert bridged.sum() == support.sum() + 2
    assert bridged[2:7, 5:7].sum() >= 2
    assert not bridged[0].any()
    assert not bridged[-1].any()


def test_joint_partition_probabilities_cover_footprint_without_thresholding() -> None:
    probabilities = torch.tensor(
        [
            [
                [
                    [0.9, 0.9, 0.9, 0.9],
                    [0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2],
                    [0.2, 0.2, 0.2, 0.2],
                ],
                [
                    [0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.8, 0.8],
                    [0.8, 0.8, 0.8, 0.8],
                    [0.8, 0.8, 0.8, 0.8],
                ],
            ]
        ]
    )
    boundary = np.asarray(
        [[0, 0], [0, 4], [4, 4], [4, 0]],
        dtype=np.float32,
    )
    rooms, _operations, adjacency = decode_mask_partition_rooms(
        torch.zeros_like(probabilities),
        boundary,
        [0, 1],
        np.asarray([0, 1]),
        torch.tensor([0.9, 0.8]),
        rplan_taxonomy(),
        canvas_size=4,
        confidence_threshold=0.5,
        minimum_area=1,
        mask_probabilities=probabilities,
    )

    assert len(rooms) == 2
    assert adjacency == [["room_01", "room_02"]]
