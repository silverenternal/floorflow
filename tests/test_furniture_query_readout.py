from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.furniture_vector_readout import read_furniture_queries


def test_furniture_budget_keeps_highest_confidence_queries() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter({"Bed": 1, "Chair": 1}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    class_logits = torch.full((2, len(taxonomy.furniture_ids) + 1), -8.0)
    class_logits[0, 0] = 1.0
    class_logits[1, 1] = 8.0
    masks = torch.full((2, 8, 8), -8.0)
    masks[0, 1:4, 1:4] = 8.0
    masks[1, 4:7, 4:7] = 8.0
    room_labels = torch.full((8, 8), taxonomy.room_ids[0])
    furniture = read_furniture_queries(
        class_logits,
        masks,
        room_labels,
        taxonomy,
        query_budget=1,
        presence_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
    )
    assert len(furniture) == 1
    assert furniture[0].label == taxonomy.class_names[
        taxonomy.furniture_ids[1]
    ]


def test_furniture_active_mask_overrides_vector_ranking() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter({"Bed": 1, "Chair": 1}),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    class_logits = torch.full((2, len(taxonomy.furniture_ids) + 1), -8.0)
    class_logits[0, 0] = 1.0
    class_logits[1, 1] = 8.0
    masks = torch.full((2, 8, 8), -100.0)
    masks[0, 1:4, 1:4] = 8.0
    masks[1, 4:7, 4:7] = 8.0
    furniture = read_furniture_queries(
        class_logits,
        masks,
        torch.full((8, 8), taxonomy.room_ids[0]),
        taxonomy,
        query_budget=1,
        active_queries=torch.tensor([[True, False]]),
        presence_threshold=0.1,
        mask_threshold=0.5,
        min_area=1,
    )
    assert len(furniture) == 1
    assert furniture[0].label == taxonomy.class_names[
        taxonomy.furniture_ids[0]
    ]
