from __future__ import annotations

from collections import Counter

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.ce2eplan_adapter import adapt_ce2eplan_row


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 1, "Kitchen": 1}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_ce2eplan_adapter_builds_labels_and_explicit_plan() -> None:
    taxonomy = _taxonomy()
    row = {
        "name": "case",
        "boundary": [
            [8, 8, 0, 0],
            [56, 8, 0, 0],
            [56, 56, 0, 0],
            [8, 56, 0, 0],
        ],
        "rooms": [
            {
                "id": 0,
                "category": 1,
                "order": 0,
                "order_aligned": 0,
                "box": [8, 8, 30, 56],
                "box_aligned": [8, 8, 30, 56],
                "r_boundary": [[8, 8], [30, 8], [30, 56], [8, 56]],
                "r_boundary_aligned": [
                    [8, 8],
                    [30, 8],
                    [30, 56],
                    [8, 56],
                ],
            },
            {
                "id": 1,
                "category": 2,
                "order": 1,
                "order_aligned": 1,
                "box": [34, 8, 56, 56],
                "box_aligned": [34, 8, 56, 56],
                "r_boundary": [[34, 8], [56, 8], [56, 56], [34, 56]],
                "r_boundary_aligned": [
                    [34, 8],
                    [56, 8],
                    [56, 56],
                    [34, 56],
                ],
            },
        ],
    }
    labels, footprint, plan, rooms = adapt_ce2eplan_row(
        row,
        taxonomy,
        prediction=True,
    )
    assert labels.shape == (256, 256)
    assert footprint[20, 20] == 1
    assert footprint[0, 0] == 0
    assert taxonomy.class_id("wall") in np.unique(labels)
    assert taxonomy.class_id("room:Bedroom") in np.unique(labels)
    assert taxonomy.class_id("room:Kitchen") in np.unique(labels)
    assert len(plan.walls) == 4
    assert len(plan.rooms) == 2
    assert len(rooms) == 2


def test_ce2eplan_category_four_is_balcony() -> None:
    row = {
        "name": "balcony",
        "boundary": [[8, 8, 0, 0], [56, 8, 0, 0], [56, 56, 0, 0]],
        "rooms": [
            {
                "id": 0,
                "category": 4,
                "order": 0,
                "r_boundary": [[8, 8], [56, 8], [56, 56], [8, 56]],
            }
        ],
    }
    taxonomy = _taxonomy()
    taxonomy = type(taxonomy)(
        class_names=(*taxonomy.class_names, "room:Balcony"),
        room_map={},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )
    _labels, _footprint, _plan, rooms = adapt_ce2eplan_row(
        row,
        taxonomy,
        prediction=False,
    )
    assert rooms[0]["label"] == "room:Balcony"
