from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.vector_scaffold_readout import vector_scaffold_plan


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10}),
            "furniture": Counter(),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_vector_scaffold_composes_wall_and_opening_vectors() -> None:
    taxonomy = _taxonomy()
    room_logits = torch.zeros(len(taxonomy.room_ids), 32, 32)
    room_logits[0] = 8.0
    footprint = torch.ones(32, 32)
    wall_endpoints = torch.tensor(
        [
            [[0.1, 0.5], [0.9, 0.5]],
            [[0.2, 0.2], [0.2, 0.8]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    wall_exists = torch.tensor([8.0, 8.0, -8.0])
    wall_thickness = torch.tensor([0.01, 0.01, 0.01])
    opening_centers = torch.tensor([[0.5, 0.5], [0.2, 0.5]])
    opening_exists = torch.tensor([8.0, -8.0])
    opening_widths = torch.tensor([0.1, 0.1])
    opening_classes = torch.zeros(2, len(taxonomy.opening_ids))
    opening_classes[:, 0] = 1.0
    opening_host_logits = torch.tensor(
        [
            [-8.0, 8.0, -8.0],
            [-8.0, -8.0, -8.0],
        ]
    )
    plan = vector_scaffold_plan(
        room_logits,
        footprint,
        wall_endpoints,
        wall_exists,
        wall_thickness,
        opening_centers,
        opening_exists,
        opening_widths,
        opening_classes,
        taxonomy,
        threshold=0.5,
        opening_host_logits=opening_host_logits,
    )
    assert len(plan.walls) == 2
    assert len(plan.openings) == 1
    assert plan.openings[0].wall_index == 1
    assert plan.rooms
