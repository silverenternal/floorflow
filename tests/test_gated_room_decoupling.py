from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.gated_room_decoupling import gated_room_decoupling


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Kitchen": 10}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_gated_decoupling_keeps_base_semantics_and_adds_confident_geometry() -> None:
    taxonomy = _taxonomy()
    bedroom = taxonomy.class_id("room:Bedroom")
    kitchen = taxonomy.class_id("room:Kitchen")
    footprint = torch.ones((1, 8, 8))
    base = torch.zeros((1, 8, 8), dtype=torch.long)
    base[:, 1:7, 1:5] = bedroom
    graph = base.clone()
    graph[:, 1:7, 5:7] = kitchen
    base_logits = torch.zeros((1, 2, 8, 8))
    base_logits[:, 0] = 3.0
    graph_logits = torch.zeros((1, 2, 8, 8))
    graph_logits[:, 1] = 5.0
    output = gated_room_decoupling(
        base,
        graph,
        base_logits,
        graph_logits,
        footprint,
        taxonomy,
        graph_confidence_threshold=0.8,
    )
    assert (output.room_labels[:, 1:7, 5:7] == bedroom).all()
    assert output.diagnostics["accepted_graph_fraction"] > 0.0
    assert output.support[0, 1:7, 1:7].all()
