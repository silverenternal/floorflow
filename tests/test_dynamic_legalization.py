from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.distribution_overlay_metrics import constraint_metrics
from floorflow.geometry.dynamic_closure import (
    apply_dynamic_wall_closure,
    closure_radius_from_geometry,
)
from floorflow.inference.dynamic_legalization import legalize_layers
from floorflow.representation.layered_floorplan import LayeredFloorplan


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Bed": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_legalize_layers_preserves_openings_and_filters_furniture() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones((1, 16, 16))
    rooms = torch.zeros((1, 16, 16), dtype=torch.long)
    rooms[:, 2:14, 2:14] = taxonomy.class_id("room:Bedroom")
    wall = torch.zeros_like(rooms)
    wall[:, 1, 2:14] = taxonomy.class_id("wall")
    opening = torch.zeros_like(rooms)
    opening[:, 1, 4] = taxonomy.class_id("door:Swing Beside")
    opening[:, 8, 8] = taxonomy.class_id("door:Swing Beside")
    furniture = torch.zeros_like(rooms)
    furniture[:, 6, 6] = taxonomy.class_id("furniture:Bed")
    furniture[:, 15, 15] = taxonomy.class_id("furniture:Bed")
    layers = LayeredFloorplan(
        room_labels=rooms,
        wall=wall,
        opening=opening,
        furniture=furniture,
        footprint=footprint,
        taxonomy=taxonomy,
    )
    legal = legalize_layers(layers, footprint, taxonomy)
    assert legal.opening[0, 1, 4] > 0
    assert legal.opening[0, 8, 8] > 0
    assert legal.furniture[0, 6, 6] > 0
    assert legal.furniture[0, 15, 15] == 0
    metrics = constraint_metrics(
        legal.visible_labels()[0],
        rooms[0],
        footprint[0, 0],
        legal.wall[0],
        legal.opening[0],
        legal.furniture[0],
        taxonomy,
        predicted_room_support=torch.isin(
            legal.room_labels[0],
            torch.tensor(taxonomy.room_ids),
        ),
    )
    assert metrics["furniture_room_containment"] == 1.0


def test_dynamic_wall_closure_uses_predicted_seam_gap() -> None:
    taxonomy = _taxonomy()
    footprint = torch.ones((1, 16, 16))
    rooms = torch.full((1, 16, 16), taxonomy.class_id("room:Bedroom"))
    rooms[:, :, 8:] = taxonomy.class_id("room:Kitchen")
    wall = torch.zeros((1, 16, 16), dtype=torch.bool)
    wall[:, :, 8] = True
    wall[:, 7:9, 8] = False
    radius = closure_radius_from_geometry(
        wall,
        rooms,
        footprint,
        max_radius=2,
    )
    closed, applied = apply_dynamic_wall_closure(
        wall,
        rooms,
        footprint,
        max_radius=2,
    )
    assert radius.item() >= 1
    assert torch.equal(radius, applied)
    assert closed[:, :, 8].all()
