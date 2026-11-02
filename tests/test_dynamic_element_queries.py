from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.element_head import SemanticElementDistributionHead
from floorflow.distribution.hierarchy import ChannelHierarchy
from floorflow.distribution.query_budget import allocate_element_query_budgets


def test_element_query_budget_tracks_predicted_scene_density() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    head = SemanticElementDistributionHead(
        8,
        hierarchy,
        hidden_channels=8,
        queries_per_group={
            "wall": 4,
            "opening": 4,
            "room": 4,
            "furniture": 4,
            "support": 4,
        },
        token_grid=4,
        output_resolution=8,
        dynamic_query_budget=True,
    )
    reference = torch.ones(2, 1, 8, 8)
    low = allocate_element_query_budgets(
        {
            "room": torch.full((2, 1, 8, 8), 0.05),
            "furniture": torch.full((2, 1, 8, 8), 0.01),
        },
        reference,
        head.query_counts,
    )
    high = allocate_element_query_budgets(
        {
            "room": torch.full((2, 1, 8, 8), 0.8),
            "furniture": torch.full((2, 1, 8, 8), 0.5),
        },
        reference,
        head.query_counts,
    )
    assert (high["furniture"] > low["furniture"]).all()
    assert (high["room"] >= low["room"]).all()
    assert (high["furniture"] <= 4).all()
