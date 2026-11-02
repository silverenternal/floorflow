from __future__ import annotations

import torch
from torch import nn

from floorflow.training.typed_channel_route_metrics import (
    typed_channel_module_gradient_metrics,
)


def test_room_module_gradient_metrics_cover_live_heads() -> None:
    model = nn.Module()
    stage = nn.Module()
    decoder = nn.Module()
    decoder.mask_head = nn.Linear(2, 1)
    decoder.query_feature = nn.Embedding(2, 2)
    stage.decoder = decoder
    model.room_instance_stage = stage
    decoder.mask_head(torch.ones(1, 2)).sum().backward()
    metrics = typed_channel_module_gradient_metrics(model)
    assert "room_mask_head_gradient_norm" in metrics
    assert "room_query_embedding_gradient_norm" not in metrics
    assert metrics["room_mask_head_gradient_norm"] > 0.0
