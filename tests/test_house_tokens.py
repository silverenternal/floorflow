import numpy as np
import torch

from floorflow.chip_flow.house_tokens import room_box_case
from floorflow.data.taxonomy import FloorplanTaxonomy


def _taxonomy() -> FloorplanTaxonomy:
    names = ("background", "wall", "room:LivingRoom", "room:Bedroom")
    return FloorplanTaxonomy(
        class_names=names,
        room_map={"LivingRoom": "room:LivingRoom", "Bedroom": "room:Bedroom"},
        furniture_map={},
        door_map={},
        window_map={},
        source_counts={},
    )


def test_room_box_case_is_normalized_and_has_ten_features() -> None:
    target = np.zeros((32, 32), dtype=np.uint8)
    target[4:16, 2:14] = 2
    target[16:28, 14:30] = 3
    footprint = (target > 0).astype(np.uint8)
    case = room_box_case(target, footprint, _taxonomy())
    assert case is not None
    assert case["boxes"].shape == (2, 4)
    assert case["features"].shape == (2, 10)
    assert torch.all(case["boxes"] >= 0)
    assert torch.all(case["boxes"][:, :2] + case["boxes"][:, 2:] <= 1.0)
