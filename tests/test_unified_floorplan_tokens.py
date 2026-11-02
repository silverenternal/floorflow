from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalFurniture,
    CanonicalOpening,
    CanonicalOperation,
    CanonicalRoom,
    CanonicalWall,
    canonical_fingerprint,
)
from floorflow.evaluation.fevp_metric_contracts import FEVPTolerances
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutFurniture,
    ReadoutOpening,
    ReadoutWall,
)
from floorflow.representation.canonical_plan_adapter import (
    canonical_plan_from_readout,
)
from floorflow.representation.unified_floorplan_tokens import (
    TOKEN_SCHEMA_VERSION,
    CoordinateQuantizer,
    FloorplanToken,
    canonicalize_floorplan,
    decode_floorplan,
    render_floorplan_rasters,
    serialize_tokens,
    tokenize_floorplan,
    validate_vector_floorplan,
)


def _taxonomy() -> FloorplanTaxonomy:
    return FloorplanTaxonomy(
        class_names=(
            "background",
            "wall",
            "door:Door",
            "window:Window",
            "room:LivingRoom",
            "room:Kitchen",
            "furniture:Chair",
        ),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_canonicalization_is_idempotent_across_junction_rounding() -> None:
    room = ((0, 0), (100, 0), (100, 100), (0, 100))
    plan = CanonicalFloorplan(
        case_id="junction-rounding",
        track="rplan_generation",
        width=100,
        height=100,
        footprint=room,
        rooms=(CanonicalRoom("room_1", "room:LivingRoom", room, room),),
        walls=(
            CanonicalWall(
                "wall_1",
                (81.06249, 145.0),
                (90.0, 145.0),
                1.0,
            ),
            CanonicalWall(
                "wall_2",
                (81.06251, 77.0),
                (90.0, 77.0),
                1.0,
            ),
        ),
        openings=(),
        furniture=(),
        adjacency=(),
        source_hash="junction-rounding",
    )
    first, _ = canonicalize_floorplan(plan, _taxonomy())
    second, _ = canonicalize_floorplan(first, _taxonomy())
    assert canonical_fingerprint(first) == canonical_fingerprint(second)


def _plan() -> CanonicalFloorplan:
    living = ((2, 2), (15, 2), (15, 17), (2, 17))
    kitchen = ((15, 2), (30, 2), (30, 17), (15, 17))
    return CanonicalFloorplan(
        case_id="vector-case",
        track="cubicasa_reconstruction",
        width=32,
        height=32,
        footprint=((1, 1), (31, 1), (31, 18), (1, 18)),
        rooms=(
            CanonicalRoom(
                "room_1",
                "room:LivingRoom",
                living,
                living,
                holes=(((5, 5), (7, 5), (7, 7), (5, 7)),),
            ),
            CanonicalRoom("room_2", "room:Kitchen", kitchen, kitchen),
        ),
        walls=(
            CanonicalWall("wall_1", (1, 1), (31, 1), 1),
            CanonicalWall("wall_2", (31, 1), (31, 18), 1),
            CanonicalWall("wall_3", (31, 18), (1, 18), 1),
            CanonicalWall("wall_4", (1, 18), (1, 1), 1),
            CanonicalWall("wall_5", (15, 2), (15, 17), 1),
        ),
        openings=(
            CanonicalOpening("opening_1", "door:Door", (15, 8), 3, "wall_5"),
        ),
        furniture=(
            CanonicalFurniture(
                "furniture_1",
                "furniture:Chair",
                ((8, 8), (10, 8), (10, 10), (8, 10)),
                "room_1",
            ),
        ),
        adjacency=(("room_1", "room_2"),),
        source_hash="source",
        operations=(
            CanonicalOperation("snap", "room_1", "moved | 1px"),
        ),
    )


def test_roundtrip_is_exact_vector_first_with_explicit_end() -> None:
    encoded = tokenize_floorplan(_plan(), _taxonomy())
    decoded = decode_floorplan(encoded.text, _taxonomy())
    assert encoded.tokens[-1] == FloorplanToken("END")
    assert encoded.tokens[0] == FloorplanToken(
        "SCHEMA",
        (TOKEN_SCHEMA_VERSION,),
    )
    assert encoded.exact_quantization
    assert canonical_fingerprint(decoded) == canonical_fingerprint(encoded.plan)
    assert decoded.rooms[0].holes
    assert decoded.operations == encoded.plan.operations
    assert decoded.junctions


def test_canonical_tokens_ignore_input_order_and_winding() -> None:
    plan = _plan()
    shuffled = replace(
        plan,
        footprint=tuple(reversed(plan.footprint)),
        rooms=tuple(
            replace(
                room,
                polygon=tuple(reversed(room.polygon)),
                support=tuple(reversed(room.support or ())),
                holes=tuple(tuple(reversed(hole)) for hole in room.holes),
            )
            for room in reversed(plan.rooms)
        ),
        walls=tuple(
            replace(wall, start=wall.end, end=wall.start)
            for wall in reversed(plan.walls)
        ),
        openings=tuple(reversed(plan.openings)),
        furniture=tuple(reversed(plan.furniture)),
        adjacency=tuple((right, left) for left, right in plan.adjacency),
        operations=tuple(reversed(plan.operations)),
    )
    left = tokenize_floorplan(plan, _taxonomy())
    right = tokenize_floorplan(shuffled, _taxonomy())
    assert left.text == right.text
    assert canonical_fingerprint(left.plan) == canonical_fingerprint(right.plan)


def test_quantization_is_reported_and_applied_before_decode() -> None:
    plan = _plan()
    room = replace(
        plan.rooms[0],
        polygon=((2.03, 2), (15, 2), (15, 17), (2, 17)),
        support=None,
    )
    encoded = tokenize_floorplan(replace(plan, rooms=(room, plan.rooms[1])), _taxonomy())
    assert not encoded.exact_quantization
    assert encoded.max_quantization_error == pytest.approx(0.03)
    decoded = decode_floorplan(encoded.tokens, _taxonomy())
    assert decoded.rooms[0].polygon[0] == (2.0, 2.0)


def test_token_stream_records_quantizer_scale() -> None:
    quantizer = CoordinateQuantizer(scale=8)
    encoded = tokenize_floorplan(
        _plan(),
        _taxonomy(),
        quantizer=quantizer,
    )
    assert "QUANTIZER|8" in encoded.text
    decoded = decode_floorplan(encoded.text, _taxonomy())
    assert canonical_fingerprint(decoded) == canonical_fingerprint(
        encoded.plan
    )


def test_delta_coordinate_tokens_are_reversible_and_compact() -> None:
    absolute = tokenize_floorplan(
        _plan(),
        _taxonomy(),
        delta_coordinates=False,
    )
    compact = tokenize_floorplan(
        _plan(),
        _taxonomy(),
        delta_coordinates=True,
    )
    assert any(token.kind == "VD" for token in compact.tokens)
    assert compact.byte_count <= absolute.byte_count
    decoded = decode_floorplan(compact.text, _taxonomy())
    assert canonical_fingerprint(decoded) == canonical_fingerprint(
        compact.plan
    )


def test_coordinate_modes_are_not_mixed() -> None:
    compact = tokenize_floorplan(_plan(), _taxonomy())
    malformed = []
    replaced = False
    for token in compact.tokens:
        if token.kind == "VD" and not replaced:
            malformed.append(
                FloorplanToken(
                    "VERTEX",
                    (str(int(token.args[0]) + 2), str(int(token.args[1]) + 2)),
                )
            )
            replaced = True
        else:
            malformed.append(token)
    assert replaced
    with pytest.raises(ValueError, match="VERTEX after the DELTA origin"):
        decode_floorplan(malformed, _taxonomy())


def test_malformed_sequences_fail_closed() -> None:
    encoded = tokenize_floorplan(_plan(), _taxonomy())
    with pytest.raises(ValueError, match="END"):
        decode_floorplan(encoded.tokens[:-1], _taxonomy())
    with pytest.raises(ValueError, match="unknown wall"):
        decode_floorplan(
            serialize_tokens(
                replace(token, args=("opening_1", "door:Door", "missing", *token.args[3:]))
                if token.kind == "OPENING"
                else token
                for token in encoded.tokens
            ),
            _taxonomy(),
        )
    duplicate = replace(
        encoded.plan,
        rooms=(encoded.plan.rooms[0], replace(encoded.plan.rooms[1], id="room_1")),
    )
    with pytest.raises(ValueError, match="room ids"):
        tokenize_floorplan(duplicate, _taxonomy())


def test_planar_validation_and_raster_diagnostics_are_explicit() -> None:
    encoded = tokenize_floorplan(_plan(), _taxonomy())
    errors = validate_vector_floorplan(
        encoded.plan,
        _taxonomy(),
        tolerances=FEVPTolerances(),
    )
    assert errors == []
    rasters = render_floorplan_rasters(encoded.plan, _taxonomy())
    assert rasters["room_overlap_pixels"] == 0.0
    assert rasters["outside_footprint_pixels"] == 0
    assert np.count_nonzero(rasters["opening_labels"]) > 0
    assert np.count_nonzero(rasters["furniture_labels"]) > 0
    assert np.count_nonzero(rasters["visible_labels"]) > 0
    room_labels = rasters["room_labels"]
    assert room_labels[6, 6] == 0

    overlapping = replace(
        encoded.plan,
        rooms=(
            encoded.plan.rooms[0],
            replace(
                encoded.plan.rooms[1],
                polygon=((14, 2), (29, 2), (29, 17), (14, 17)),
                support=((14, 2), (29, 2), (29, 17), (14, 17)),
            ),
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        render_floorplan_rasters(overlapping, _taxonomy())


def test_raster_preserves_hole_free_vector_partition_under_rounding() -> None:
    living = (
        (0, 0),
        (4, 0),
        (4, 1),
        (4, 8),
        (4.2, 8),
        (4.2, 10),
        (0, 10),
    )
    kitchen = (
        (4, 0),
        (10, 0),
        (10, 10),
        (4.3, 10),
        (4.3, 8),
        (5.5, 8),
        (5.5, 1),
        (4, 1),
    )
    plan = CanonicalFloorplan(
        case_id="raster-topology",
        track="rplan_generation",
        width=12,
        height=12,
        footprint=((0, 0), (10, 0), (10, 10), (0, 10)),
        rooms=(
            CanonicalRoom("room_1", "room:LivingRoom", living, living),
            CanonicalRoom("room_2", "room:Kitchen", kitchen, kitchen),
        ),
        walls=(),
        openings=(),
        furniture=(),
        adjacency=(("room_1", "room_2"),),
        source_hash="raster-topology",
    )

    labels = render_floorplan_rasters(
        plan,
        _taxonomy(),
        strict=False,
    )["room_labels"]
    occupancy = np.isin(labels, tuple(_taxonomy().room_ids))

    assert not (binary_fill_holes(occupancy) & ~occupancy).any()


def test_existing_readout_plan_enters_the_same_token_contract() -> None:
    plan = _plan()
    readout = RasterWallPlan(
        frame=32,
        walls=[
            ReadoutWall(wall.start, wall.end, wall.thickness, wall.wall_type)
            for wall in plan.walls
        ],
        rooms=[
            {
                "label": room.label,
                "polygon": room.polygon,
                "holes": room.holes,
            }
            for room in plan.rooms
        ],
        openings=[
            ReadoutOpening((15, 8), 3, "door:Door", 4),
        ],
        furniture=[
            ReadoutFurniture(
                "furniture:Chair",
                plan.furniture[0].polygon,
                0,
            )
        ],
    )
    canonical = canonical_plan_from_readout(
        readout,
        _taxonomy(),
        case_id="adapter-case",
        track="typed_editing",
        width=32,
        height=32,
        footprint=plan.footprint,
        source_hash="adapter-source",
    )
    encoded = tokenize_floorplan(canonical, _taxonomy())
    decoded = decode_floorplan(encoded.tokens, _taxonomy())
    assert decoded.openings[0].wall_id == "wall_5"
    assert decoded.furniture[0].room_id == "room_1"
    assert decoded.adjacency == (("room_1", "room_2"),)
