from __future__ import annotations

import pytest
import torch

from floorflow.heads.channel_reliability import (
    reliability_weighted_residual,
)


def test_reliability_downweights_uncertain_residual_without_hard_mask() -> None:
    residual = torch.ones(1, 2, 4, 4)
    reliability = torch.zeros(1, 1, 4, 4)
    reliability[:, :, :, :2] = 1.0
    weighted = reliability_weighted_residual(residual, reliability)
    assert torch.allclose(weighted[:, :, :, :2], torch.ones(1, 2, 4, 2))
    assert torch.allclose(weighted[:, :, :, 2:], torch.full((1, 2, 4, 2), 0.75))
    with pytest.raises(ValueError, match="reliability"):
        reliability_weighted_residual(residual, torch.zeros(1, 3, 4, 4))
