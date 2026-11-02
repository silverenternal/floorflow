from __future__ import annotations

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    rplan_taxonomy,
)
from floorflow.evaluation.rplan_target_contract import (
    audit_rplan_target_item,
)


def _row() -> dict:
    return {
        "name": "contract",
        "boundary": [[10, 10], [10, 90], [90, 90], [90, 10]],
        "rooms": [
            {
                "id": 0,
                "category": 0,
                "location": [40, 50],
                "r_boundary": [[10, 10], [50, 10], [50, 90], [10, 90]],
            }
        ],
        "adjacencies": [],
    }


def test_rplan_target_contract_accepts_adapter_output() -> None:
    dataset = RPlanBoundaryDataset(
        "unused.json",
        rplan_taxonomy(),
        rows=[_row()],
        limits=RPlanTensorLimits(
            image_size=32,
            canvas_size=100,
            max_boundary_points=8,
            max_rooms=4,
            room_points=12,
            max_walls=16,
            max_openings=2,
        ),
    )
    result = audit_rplan_target_item(dataset[0])
    assert result["passed"]
    assert result["violations"] == ()


def test_rplan_target_contract_detects_invalid_host_and_room_prefix() -> None:
    dataset = RPlanBoundaryDataset(
        "unused.json",
        rplan_taxonomy(),
        rows=[_row()],
        limits=RPlanTensorLimits(
            image_size=32,
            canvas_size=100,
            max_boundary_points=8,
            max_rooms=4,
            room_points=12,
            max_walls=16,
            max_openings=2,
        ),
    )
    item = dataset[0]
    item["opening_valid"][0] = True
    item["opening_host"][0] = -100
    item["room_point_valid"][0, -1] = True
    result = audit_rplan_target_item(item)
    assert "opening_invalid_host" in result["violations"]
    assert "room_point_valid_nonprefix" in result["violations"]
