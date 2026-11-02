from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.inference.geometry_wall import geometry_wall_mask


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 4}),
            "furniture": Counter({"Sink": 4}),
            "door": Counter({"Swing Beside": 4}),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_geometry_wall_readout_adds_only_high_confidence_walls() -> None:
    taxonomy = _taxonomy()
    height = width = 16
    room_logits = torch.full((1, 1, height, width), -4.0)
    wall_logits = torch.full((1, 1, height, width), -8.0)
    wall_logits[:, :, :, 6:9] = 8.0
    zeros = torch.zeros(1, 16, 16)
    auxiliary = {
        "room_partition_logits": room_logits,
        "structured_objects": {
            "opening_logits": torch.zeros(1, len(taxonomy.opening_ids), 16, 16),
            "opening_background_logits": zeros,
            "furniture_logits": torch.zeros(1, len(taxonomy.furniture_ids), 16, 16),
            "furniture_background_logits": zeros,
            "wall_logits": torch.full((1, 16, 16), -4.0),
        },
        "geometry": {"wall_centerline_logits": wall_logits},
        "semantic_logits": None,
        "hierarchical_rooms": None,
        "polygon_loops": None,
        "room_mask_predictions": None,
    }

    class FakeModel:
        target_channels = 2

        def forward_with_aux(self, *args, **kwargs):
            return torch.zeros(1, 2, 16, 16), auxiliary

    labels = decode_structured_floorplan(
        FakeModel(),
        torch.zeros(1, 2, 16, 16),
        torch.ones(1, 16, 16),
        taxonomy,
        opening_threshold=0.5,
        furniture_threshold=0.5,
        wall_threshold=0.5,
        object_score_scale="probability",
        structural_gate=False,
        room_source="partition",
        use_geometry_walls=True,
        geometry_wall_threshold=0.7,
        geometry_wall_dilate=1,
    )
    wall_id = taxonomy.class_id("wall")
    assert (labels[0, :, 7] == wall_id).all()
    assert not (labels[0, :, 1] == wall_id).any()


def test_geometry_wall_branch_returns_thresholded_mask() -> None:
    class FakeModel:
        target_channels = 2

        def forward_with_aux(self, *args, **kwargs):
            logits = torch.full((1, 1, 8, 8), -8.0)
            logits[:, :, :, 3:5] = 8.0
            return torch.zeros(1, 2, 8, 8), {"geometry": {"wall_centerline_logits": logits}}

    wall = geometry_wall_mask(FakeModel(), torch.ones(1, 8, 8), threshold=0.7)
    assert wall is not None
    assert wall[:, :, 3:5].all()
    assert not wall[:, :, 0].any()
