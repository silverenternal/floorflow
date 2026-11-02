from __future__ import annotations

import pytest
import torch


def test_cached_pretrained_vit_exposes_patch_map() -> None:
    pytest.importorskip("timm")
    from floorflow.model.pretrained_vit import FrozenViTPatchFeatures

    adapter = FrozenViTPatchFeatures(image_size=256)
    output = adapter(torch.ones(1, 1, 256, 256))
    assert output.ndim == 4
    assert output.shape[1] == adapter.channels
    assert not any(parameter.requires_grad for parameter in adapter.parameters())
