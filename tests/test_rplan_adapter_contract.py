from __future__ import annotations

import numpy as np
import pytest
import torch

from floorflow.data.rplan_boundary_adapter import (
    RPlanBoundaryDataset,
    RPlanTensorLimits,
    collate_rplan_items,
    entrance_segment_from_row,
    limit_canonical_plan_rooms,
    rplan_taxonomy,
)
from floorflow.data.rplan_model_limits import validate_rplan_model_limits


def _row() -> dict:
    return {
        "name": "synthetic-rplan",
        "boundary": [
            [10, 10],
            [10, 90],
            [90, 90],
            [90, 10],
        ],
        "entrance_expand": [
            [10, 10],
            [18, 10],
            [18, 24],
            [10, 24],
        ],
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


def _limits(*, max_walls: int = 16) -> RPlanTensorLimits:
    return RPlanTensorLimits(
        image_size=32,
        canvas_size=100,
        max_boundary_points=8,
        max_rooms=4,
        room_points=12,
        max_walls=max_walls,
        max_openings=2,
    )


def test_rplan_tensorization_resamples_instead_of_truncating_complex_rings() -> None:
    taxonomy = rplan_taxonomy()
    boundary = [
        [10, 10],
        *[[10, value] for value in np.linspace(20, 80, 8)],
        [50, 90],
        *[[90, value] for value in np.linspace(80, 20, 8)],
        [90, 10],
    ]
    room_boundary = [
        *[[20, value] for value in np.linspace(20, 80, 8)],
        *[[80, value] for value in np.linspace(80, 20, 8)],
    ]
    row = {
        "name": "complex-rplan",
        "boundary": boundary,
        "rooms": [
            {
                "id": 0,
                "category": 0,
                "location": [50, 50],
                "r_boundary": room_boundary,
            }
        ],
        "adjacencies": [],
    }
    dataset = RPlanBoundaryDataset(
        "unused.json",
        taxonomy,
        rows=[row],
        limits=RPlanTensorLimits(
            image_size=32,
            canvas_size=100,
            max_boundary_points=8,
            max_rooms=2,
            room_points=12,
        ),
    )
    item = dataset[0]
    assert item["boundary_valid"].all()
    assert item["room_point_valid"][0].all()


def test_rplan_opening_without_retained_host_is_invalid() -> None:
    dataset = RPlanBoundaryDataset(
        "unused.json",
        rplan_taxonomy(),
        rows=[_row()],
        limits=_limits(max_walls=0),
    )
    item = dataset[0]
    assert not item["opening_valid"].any()
    assert (item["opening_host"] < 0).all()


def test_rplan_verifier_key_binds_geometry_not_only_tokens() -> None:
    dataset = RPlanBoundaryDataset(
        "unused.json",
        rplan_taxonomy(),
        rows=[_row()],
        limits=_limits(),
    )
    first = dataset[0]
    second = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in first.items()
    }
    second["wall_endpoints"][0, 0, 0] += 0.01
    keyed = collate_rplan_items([first, second])
    assert (
        keyed["verifier_target_keys"][0]
        != keyed["verifier_target_keys"][1]
    )


def test_rplan_collate_rejects_inconsistent_grammar_vocabulary() -> None:
    first = {
        "case_name": "first",
        "token_ids": torch.tensor([1]),
        "token_mask": torch.tensor([True]),
        "token_loss_mask": torch.tensor([True]),
        "token_stage_ids": torch.tensor([0]),
        "token_kind_ids": torch.tensor([0]),
        "token_grammar_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    second = {
        **first,
        "case_name": "second",
        "token_grammar_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="vocabulary size mismatch"):
        collate_rplan_items([first, second])


def test_rplan_room_targets_share_one_retained_room_set() -> None:
    row = _row()
    row["rooms"].append(
        {
            "id": 1,
            "category": 1,
            "location": [70, 50],
            "r_boundary": [[50, 10], [90, 10], [90, 90], [50, 90]],
        }
    )
    row["adjacencies"] = [[0, 1]]
    dataset = RPlanBoundaryDataset(
        "unused.json",
        rplan_taxonomy(),
        rows=[row],
        limits=RPlanTensorLimits(
            image_size=32,
            canvas_size=100,
            max_boundary_points=8,
            max_rooms=1,
            room_points=12,
            max_walls=16,
            max_openings=2,
        ),
    )
    item = dataset[0]
    assert int(item["room_exists"].sum()) == 1
    assert int(item["room_count"]) == 1
    assert int(item["source_room_count"]) == 2
    assert item["adjacency"].shape == (1, 1)
    limited = limit_canonical_plan_rooms(dataset._plan(0), 1)
    assert len(limited.rooms) == 1
    assert limited.adjacency == ()


def test_entrance_host_prefers_parallel_boundary_over_closer_side() -> None:
    segment, valid = entrance_segment_from_row(
        {
            "boundary": [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            "entrance_expand": [
                [8.2, 0.8],
                [10.2, 0.8],
                [10.2, 1.2],
                [8.2, 1.2],
            ],
        },
        canvas_size=11,
    )
    assert valid
    assert np.allclose(segment[:, 1], 0.0)
    assert not np.allclose(segment[:, 0], segment[0, 0])


def test_rplan_model_capacity_must_cover_target_limits() -> None:
    limits = _limits()
    model_config = {
        "identity_queries": limits.max_rooms,
        "max_walls": limits.max_walls,
        "max_openings": limits.max_openings,
        "polygon_points": limits.room_points,
        "max_boundary_points": limits.max_boundary_points,
    }
    validate_rplan_model_limits(limits, model_config)
    model_config["max_openings"] = limits.max_openings - 1
    with pytest.raises(ValueError, match="max_openings"):
        validate_rplan_model_limits(limits, model_config)
