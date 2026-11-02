from collections import Counter

import torch

from floorflow.data.taxonomy import build_taxonomy
from floorflow.model.cad_layers import CadLayeredOutput


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 10, "Bath": 10}),
            "furniture": Counter({"Bed": 5}),
            "door": Counter({"Swing Beside": 5}),
            "window": Counter({"Regular": 5}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def test_cad_layers_compose_without_mixing_family_thresholds() -> None:
    taxonomy = _taxonomy()
    layers = CadLayeredOutput(
        room_logits=torch.zeros(1, len(taxonomy.room_ids), 16, 16),
        wall_logits=torch.full((1, 1, 16, 16), -8.0),
        opening_logits=torch.zeros(1, len(taxonomy.opening_ids), 16, 16),
        furniture_logits=torch.zeros(1, len(taxonomy.furniture_ids), 16, 16),
        support_logits=torch.full((1, 1, 16, 16), -8.0),
    )
    layers.room_logits[:, 0] = 8.0
    layers.wall_logits[:, :, 7:9, :] = 8.0
    layers.opening_logits[:, 0, 7:9, 3:5] = 8.0
    layers.furniture_logits[:, 0, 11:14, 11:14] = 8.0
    footprint = torch.ones(1, 16, 16)
    rendered = layers.render(footprint, taxonomy)
    visible = rendered.visible_labels()
    assert rendered.room_labels[0, 0, 0].item() == taxonomy.room_ids[0]
    assert (visible[0, 7:9, :3] == taxonomy.class_id("wall")).all()
    assert visible[0, 7, 3].item() == taxonomy.opening_ids[0]
    assert visible[0, 12, 12].item() == taxonomy.furniture_ids[0]
    scattered = layers.scatter(taxonomy)
    assert scattered.shape == (
        1,
        taxonomy.num_classes,
        16,
        16,
    )


def test_cad_layer_resize_preserves_contract() -> None:
    taxonomy = _taxonomy()
    layers = CadLayeredOutput(
        room_logits=torch.zeros(1, len(taxonomy.room_ids), 8, 8),
        wall_logits=torch.zeros(1, 1, 8, 8),
        opening_logits=torch.zeros(1, len(taxonomy.opening_ids), 8, 8),
        furniture_logits=torch.zeros(1, len(taxonomy.furniture_ids), 8, 8),
    )
    resized = layers.resized((16, 16))
    assert resized.room_logits.shape[-2:] == (16, 16)
    assert resized.support_logits is None
