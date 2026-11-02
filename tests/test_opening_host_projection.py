from __future__ import annotations

from collections import Counter

import pytest
import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.opening_host_projection import (
    project_routed_openings_to_final_walls,
)


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _predictions(host: int) -> dict[str, torch.Tensor]:
    return {
        "presence_logits": torch.tensor([[4.0]]),
        "opening_mask": torch.tensor([[True]]),
        "host_index": torch.tensor([[host]]),
        "host_endpoints": torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
        "center_fraction": torch.tensor([[0.5]]),
        "width_fraction": torch.tensor([[0.2]]),
        "center_xy": torch.tensor([[[0.5, 0.0]]]),
        "type_logits": torch.tensor([[[4.0, 0.0]]]),
    }


def test_routed_host_maps_to_final_selected_wall_order() -> None:
    rows = project_routed_openings_to_final_walls(
        _predictions(host=2),
        torch.tensor(
            [[[[0.0, 0.0], [1.0, 0.0]],
              [[0.0, 0.5], [1.0, 0.5]],
              [[0.0, 1.0], [1.0, 1.0]]]]
        ),
        torch.tensor([[False, False, True]]),
        _taxonomy(),
        frame=100.0,
    )
    assert len(rows[0]) == 1
    assert rows[0][0].wall_index == 0
    assert rows[0][0].center == (50.0, 0.0)


def test_active_opening_on_unselected_host_fails_closed() -> None:
    with pytest.raises(ValueError, match="not in the final wall set"):
        project_routed_openings_to_final_walls(
            _predictions(host=1),
            torch.zeros(1, 3, 2, 2),
            torch.tensor([[True, False, False]]),
            _taxonomy(),
            frame=100.0,
        )


def test_no_selected_walls_returns_an_empty_opening_layer() -> None:
    rows = project_routed_openings_to_final_walls(
        _predictions(host=0),
        torch.zeros(1, 2, 2, 2),
        torch.zeros(1, 2, dtype=torch.bool),
        _taxonomy(),
        frame=100.0,
    )
    assert rows == [[]]
