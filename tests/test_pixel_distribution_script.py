from __future__ import annotations

import torch

from floorflow.heads.pixel_distribution import PixelDistributionHead


def test_distribution_head_can_be_used_as_standalone_classifier() -> None:
    head = PixelDistributionHead(8, 5, hidden_channels=12, room_ids=(2, 3))
    output = head(torch.randn(1, 8, 16, 16))
    labels = output["logits"].argmax(dim=1)
    assert labels.shape == (1, 16, 16)
    assert output["room_logits"].shape == (1, 2, 16, 16)
