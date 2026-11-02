from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.external_program_adapter import (
    extract_oracle_rooms,
    plan_to_housegan_payload,
)


def test_cubicasa_rooms_become_adjacent_housegan_program() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"LivingRoom": 10, "Bedroom": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    target = np.zeros((32, 32), dtype=np.uint8)
    living = taxonomy.class_id("room:LivingRoom")
    bedroom = taxonomy.class_id("room:Bedroom")
    target[4:24, 4:16] = living
    target[4:24, 16:28] = bedroom

    rooms = extract_oracle_rooms(target, taxonomy, min_area=8)
    payload = plan_to_housegan_payload(rooms, tolerance=0.03)

    assert len(rooms) == 2
    assert payload["room_type"] == [1, 3]
    assert len(payload["boxes"]) == 2
    assert len(payload["edges"]) == 8
    assert any(len(edge_rooms) == 2 for edge_rooms in payload["ed_rm"])
    assert payload["oracle_room_classes"] == ["room:LivingRoom", "room:Bedroom"]


def test_rooms_of_the_same_class_remain_separate_components() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    bedroom = taxonomy.class_id("room:Bedroom")
    target = np.zeros((32, 32), dtype=np.uint8)
    target[2:10, 2:10] = bedroom
    target[20:30, 20:30] = bedroom

    rooms = extract_oracle_rooms(target, taxonomy, min_area=8)

    assert len(rooms) == 2
    assert all(room.house_type == 3 for room in rooms)
