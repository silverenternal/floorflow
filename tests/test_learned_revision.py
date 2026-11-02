from __future__ import annotations

import numpy as np
import torch

from floorflow.inference.learned_revision import consume_learned_revision


def test_learned_revision_state_controls_typed_edit_graph() -> None:
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:4] = 1
    labels[4:] = 2
    edit_region = np.zeros((8, 8), dtype=bool)
    edit_region[2:6, 2:6] = True
    revision = {
        "family_logits": torch.tensor([[4.0, 3.0, -4.0, -4.0]]),
        "region_logits": torch.full((1, 1, 8, 8), 4.0),
        "revision_gate": torch.tensor([0.9]),
    }
    graph, families = consume_learned_revision(
        labels,
        edit_region,
        labels == 2,
        revision,
        room_ids=(1, 2),
        wall_id=3,
        opening_ids=(4,),
        furniture_ids=(5,),
    )
    assert set(families) == {"room", "wall"}
    assert graph.anchor.any()
    assert graph.changed_families == ("room", "wall")


def test_low_revision_gate_falls_back_to_declared_edit_region() -> None:
    labels = np.zeros((8, 8), dtype=np.int64)
    labels[:4] = 1
    labels[4:] = 2
    edit_region = np.zeros((8, 8), dtype=bool)
    edit_region[2:6, 2:6] = True
    revision = {
        "family_logits": torch.tensor([[4.0, 3.0, -4.0, -4.0]]),
        "region_logits": torch.full((1, 1, 8, 8), 4.0),
        "revision_gate": torch.tensor([0.1]),
    }
    graph, families = consume_learned_revision(
        labels,
        edit_region,
        labels == 2,
        revision,
        room_ids=(1, 2),
        wall_id=3,
        opening_ids=(4,),
        furniture_ids=(5,),
    )
    assert families == ("room",)
    assert graph.changed_families == ("room",)
    assert graph.anchor[4:6, 2:6].any()
