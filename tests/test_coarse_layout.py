from __future__ import annotations

from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.representation.coarse_layout import coarse_room_layout_logits


def test_coarse_layout_preserves_room_blocks() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing Beside": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, :, :8] = taxonomy.room_ids[0]
    labels[:, :, 8:] = taxonomy.room_ids[1]
    logits = coarse_room_layout_logits(labels, taxonomy, size=4)
    assert logits.shape == (1, 2, 16, 16)
    left = int(logits[0, :, 4, 2].argmax())
    right = int(logits[0, :, 4, 12].argmax())
    assert left == 0
    assert right == 1


def test_coarse_layout_refiner_accepts_validity_mask() -> None:
    taxonomy = build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter({"Sink": 2}),
            "door": Counter({"Swing Beside": 2}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )
    model = CoarseLayoutRefiner(
        len(taxonomy.room_ids),
        base_channels=8,
        masked_input=True,
    )
    coarse = torch.zeros(1, len(taxonomy.room_ids), 8, 8)
    footprint = torch.ones(1, 16, 16)
    mask = torch.ones(1, 8, 8)
    mask[:, 2:4, 2:4] = 0
    output = model(coarse, footprint, mask)
    assert output.shape == (1, len(taxonomy.room_ids), 16, 16)
