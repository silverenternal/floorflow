from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.topology import TopologyAuxHeads, derive_topology_targets, topology_aux_loss


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Kitchen": 10}),
            "furniture": Counter({"Sink": 10}),
            "door": Counter({"Swing Beside": 10}),
            "window": Counter({"Regular": 10}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_topology_targets_and_heads() -> None:
    taxonomy = _taxonomy()
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    footprint = torch.ones(1, 32, 32)
    target[:, 6:26, 6:26] = taxonomy.class_id("room:Kitchen")
    target[:, 6:9, 6:26] = taxonomy.class_id("wall")
    target[:, 6:9, 14:18] = taxonomy.class_id("door:Swing Beside")
    target[:, 16:20, 16:20] = taxonomy.class_id("furniture:Sink")
    targets = derive_topology_targets(target, footprint, taxonomy)
    assert set(targets) == {
        "wall",
        "wall_direction_x",
        "wall_direction_y",
        "opening_contact",
        "containment",
        "room_boundary",
        "adjacency",
    }
    assert targets["opening_contact"].sum() > 0
    assert targets["containment"].sum() > 0

    predictions = {
        name: torch.randn_like(value, requires_grad=True) for name, value in targets.items()
    }
    loss, components = topology_aux_loss(predictions, target, footprint, taxonomy)
    loss.backward()
    assert torch.isfinite(loss)
    assert "topology_loss" in components
    assert TopologyAuxHeads(8, len(taxonomy.room_ids))(torch.randn(1, 8, 32, 32)).keys() == targets.keys()
