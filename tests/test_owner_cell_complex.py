"""Tests for shared owner-boundary and junction cell extraction."""

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.structural_legality import wall_junction_summary
from floorflow.geometry.owner_cell_complex import (
    extract_owner_cell_complex,
)
from floorflow.representation.vector_floorplan_assembly import (
    assemble_vector_floorplan,
)


def test_shared_boundary_is_one_owner_paired_edge() -> None:
    owner = np.zeros((4, 4), dtype=np.int64)
    owner[:, 2:] = 1
    support = np.ones((4, 4), dtype=bool)

    complex_ = extract_owner_cell_complex(
        owner,
        support,
        {0: "room_01", 1: "room_02"},
    )

    assert complex_.adjacency == (("room_01", "room_02"),)
    interior = [
        edge for edge in complex_.boundaries if edge.edge_type == "interior"
    ]
    assert len(interior) == 1
    edge = interior[0]
    assert edge.start == (2.0, 0.0)
    assert edge.end == (2.0, 4.0)
    assert edge.owner_pair == (0, 1)
    assert edge.edge_type == "interior"


def test_t_junction_records_all_incident_boundaries() -> None:
    owner = np.zeros((4, 4), dtype=np.int64)
    owner[:2, 2:] = 1
    owner[2:, 2:] = 2
    support = np.ones((4, 4), dtype=bool)

    complex_ = extract_owner_cell_complex(
        owner,
        support,
        {0: "room_01", 1: "room_02", 2: "room_03"},
    )

    junction = next(
        item
        for item in complex_.junctions
        if np.allclose(item.point, (2.0, 2.0))
    )
    assert junction.degree == 3
    assert len(junction.boundary_ids) == 3
    assert set(complex_.adjacency) == {
        ("room_01", "room_02"),
        ("room_01", "room_03"),
        ("room_02", "room_03"),
    }
    summary = wall_junction_summary(
        complex_.wall_candidates(minimum_length=1.0),
        tolerance=0.1,
    )
    assert summary["open_endpoint_count"] == 0


def test_unknown_owner_pixels_are_void_not_shared_rooms() -> None:
    owner = np.zeros((3, 4), dtype=np.int64)
    owner[:, 2:] = 9
    support = np.ones((3, 4), dtype=bool)

    complex_ = extract_owner_cell_complex(
        owner,
        support,
        {0: "room_01"},
    )

    assert complex_.adjacency == ()
    assert not any(
        edge.edge_type == "interior" for edge in complex_.boundaries
    )
    edge = next(
        edge
        for edge in complex_.boundaries
        if edge.start == (2.0, 0.0) and edge.end == (2.0, 3.0)
    )
    assert edge.edge_type == "void"
    assert edge.owner_pair == (-1, 0)


def test_owner_confidence_is_carried_to_shared_wall_candidate() -> None:
    owner = np.zeros((3, 4), dtype=np.int64)
    owner[:, 2:] = 1
    support = np.ones((3, 4), dtype=bool)
    confidence = np.zeros((3, 4), dtype=np.float32)
    confidence[:, 1] = 0.4
    confidence[:, 2] = 0.8

    complex_ = extract_owner_cell_complex(
        owner,
        support,
        {0: "room_01", 1: "room_02"},
        owner_confidence=confidence,
    )
    wall = complex_.wall_candidates(minimum_length=2.0)[0]

    assert wall.wall_type == "interior"
    assert np.isclose(wall.continuity, 0.6)
    assert complex_.coverage()["owner_pair_boundary_rate"] == 1.0


def test_canonical_assembly_consumes_shared_owner_walls() -> None:
    owner = np.zeros((6, 6), dtype=np.int64)
    owner[:, 3:] = 1
    support = np.ones((6, 6), dtype=bool)
    complex_ = extract_owner_cell_complex(
        owner,
        support,
        {0: "room_01", 1: "room_02"},
    )
    taxonomy = FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Door",
            "window:Window",
            "room:LivingRoom",
            "room:Kitchen",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )

    plan = assemble_vector_floorplan(
        case_id="shared-owner",
        track="rplan_generation",
        width=6,
        height=6,
        footprint=((0, 0), (6, 0), (6, 6), (0, 6)),
        rooms=(
            {
                "id": "room_01",
                "label": "room:LivingRoom",
                "polygon": ((0, 0), (3, 0), (3, 6), (0, 6)),
            },
            {
                "id": "room_02",
                "label": "room:Kitchen",
                "polygon": ((3, 0), (6, 0), (6, 6), (3, 6)),
            },
        ),
        source_hash="shared-owner",
        taxonomy=taxonomy,
        owner_cell_complex=complex_,
        wall_source="shared_owners",
        include_entrance=False,
    )

    assert plan.adjacency == (("room_01", "room_02"),)
    interior = [
        wall for wall in plan.walls if wall.wall_type == "interior"
    ]
    assert len(interior) == 1
    assert interior[0].start == (3.0, 0.0)
    assert interior[0].end == (3.0, 6.0)
    assert sum(wall.wall_type == "exterior" for wall in plan.walls) == 6
    assert "shared_owner_wall_projection" in {
        operation.kind for operation in plan.operations
    }


def test_canonical_assembly_defaults_to_owner_cells_and_preserves_opening() -> None:
    owner = np.zeros((6, 6), dtype=np.int64)
    owner[:, 3:] = 1
    complex_ = extract_owner_cell_complex(
        owner,
        np.ones((6, 6), dtype=bool),
        {0: "room_01", 1: "room_02"},
    )
    taxonomy = FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Door",
            "window:Window",
            "room:LivingRoom",
            "room:Kitchen",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )
    plan = assemble_vector_floorplan(
        case_id="owner-default",
        track="rplan_generation",
        width=6,
        height=6,
        footprint=((0, 0), (6, 0), (6, 6), (0, 6)),
        rooms=(
            {
                "id": "room_01",
                "label": "room:LivingRoom",
                "polygon": ((0, 0), (3, 0), (3, 6), (0, 6)),
            },
            {
                "id": "room_02",
                "label": "room:Kitchen",
                "polygon": ((3, 0), (6, 0), (6, 6), (3, 6)),
            },
        ),
        source_hash="owner-default",
        taxonomy=taxonomy,
        owner_cell_complex=complex_,
        openings=(
            {
                "label": "door:Door",
                "center": (3, 3),
                "width": 1,
                "wall_index": 0,
            },
        ),
        include_entrance=False,
    )
    assert plan.openings
    assert plan.openings[0].wall_id in {wall.id for wall in plan.walls}
    assert "owner_cell_wall_projection" in {
        operation.kind for operation in plan.operations
    }
