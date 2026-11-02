from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.distribution.element_head import SemanticElementDistributionHead
from floorflow.distribution.hierarchy import ChannelHierarchy


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_element_head_outputs_hierarchical_queries() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    head = SemanticElementDistributionHead(
        12,
        hierarchy,
        hidden_channels=16,
        queries_per_group={
            "wall": 2,
            "opening": 2,
            "room": 2,
            "furniture": 2,
            "support": 1,
        },
        token_grid=4,
        output_resolution=16,
    )
    features = torch.randn(1, 12, 16, 16)
    probability = torch.softmax(
        torch.randn(1, taxonomy.num_classes, 16, 16),
        dim=1,
    )
    family_gates = {
        "wall": torch.ones(1, 1, 16, 16) * 0.3,
        "room": torch.ones(1, 1, 16, 16) * 0.5,
    }
    output = head(
        features,
        {"probability": probability, "family_gates": family_gates},
    )
    assert set(output) == {"wall", "opening", "room", "furniture", "support"}
    assert output["room"]["mask_logits"].shape == (1, 2, 16, 16)
    assert output["furniture"]["type_logits"].shape[1] == 2
    assert output["support"]["parent_gate"].shape == (1, 1)
    loss = output["room"]["mask_logits"].mean()
    loss.backward()
    assert head.group_queries["room"].grad is not None


def test_v2_masks_start_sparse_and_query_specific() -> None:
    taxonomy = _taxonomy()
    hierarchy = ChannelHierarchy.from_taxonomy(taxonomy)
    torch.manual_seed(7)
    head = SemanticElementDistributionHead(
        12,
        hierarchy,
        hidden_channels=24,
        queries_per_group={
            "wall": 4,
            "opening": 4,
            "room": 4,
            "furniture": 4,
            "support": 2,
        },
        token_grid=4,
        output_resolution=16,
        mask_version="v2",
    )
    footprint = torch.zeros(2, 1, 16, 16)
    footprint[:, :, 2:14, 2:14] = 1.0
    probability = torch.softmax(
        torch.randn(2, taxonomy.num_classes, 16, 16),
        dim=1,
    )
    output = head(
        torch.randn(2, 12, 16, 16),
        {
            "probability": probability,
            "family_gates": {
                "wall": torch.full((2, 1, 16, 16), 0.2),
                "room": torch.full((2, 1, 16, 16), 0.6),
            }
        },
        footprint=footprint,
    )
    for prediction in output.values():
        logits = prediction["mask_logits"]
        outside = (~footprint.bool()).expand_as(logits)
        assert bool((logits.masked_select(outside) <= -19.0).all())
        assert prediction["mask_mean_probability"] < 0.3
        assert prediction["mask_positive_fraction"] < 0.1
        assert prediction["mask_pairwise_iou"] < 0.25
