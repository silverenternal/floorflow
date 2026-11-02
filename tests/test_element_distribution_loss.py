from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.element_head import SemanticElementDistributionHead
from floorflow.distribution.element_loss import element_set_loss
from floorflow.distribution.element_targets import element_targets
from floorflow.distribution.hierarchy import ChannelHierarchy


def test_element_targets_and_set_loss_backpropagate() -> None:
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
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 2:8, 2:8] = taxonomy.room_ids[0]
    labels[:, 9:14, 9:14] = taxonomy.room_ids[1]
    labels[:, 6:10, 2:4] = taxonomy.class_id("wall")
    labels[:, 7:9, 7:8] = taxonomy.furniture_ids[0]
    targets = element_targets(
        labels,
        taxonomy,
        hierarchy,
        max_queries={
            "wall": 2,
            "opening": 2,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
        min_area=1,
    )
    head = SemanticElementDistributionHead(
        8,
        hierarchy,
        hidden_channels=8,
        queries_per_group={
            "wall": 2,
            "opening": 2,
            "room": 2,
            "furniture": 1,
            "support": 1,
        },
        token_grid=4,
        output_resolution=16,
    )
    predictions = head(
        torch.randn(1, 8, 16, 16),
        {
            "probability": torch.softmax(
                torch.randn(1, taxonomy.num_classes, 16, 16),
                dim=1,
            ),
            "family_gates": {},
        },
    )
    loss, components = element_set_loss(predictions, targets)
    assert "element_set_loss" in components
    assert "room_element_presence_loss" in components
    assert "room_element_mask_diversity" in components
    loss.backward()
    assert head.group_queries["room"].grad is not None
