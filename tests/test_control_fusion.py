from __future__ import annotations

import numpy as np
import torch

from floorflow.data.control_fusion import (
    fuse_controls,
    inpaint_controls,
    nearest_fill_batch,
)


def test_fuse_controls_only_replaces_invalid_regions() -> None:
    primary = {
        "room_logits": torch.zeros(2, 4, 4),
        "program": torch.tensor([True, False]),
        "structure": torch.zeros(3, 4, 4),
        "furniture": torch.zeros(3, 4, 4),
        "support": torch.zeros(3, 4, 4),
    }
    fallback = {
        key: torch.ones_like(value)
        for key, value in primary.items()
        if key != "program"
    }
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    masks = {
        f"valid_{key}": mask
        for key in ("room_logits", "structure", "furniture", "support")
    }
    fused = fuse_controls(primary, fallback, masks)
    for key in ("room_logits", "structure", "furniture", "support"):
        assert float(fused[key][0, 0, 0]) == 1.0
        assert float(fused[key][0, 1, 1]) == 0.0
    assert torch.equal(fused["program"], primary["program"])


def test_inpaint_controls_uses_nearest_valid_value() -> None:
    controls = {
        "room_logits": torch.zeros(2, 5, 5),
        "program": torch.tensor([True, False]),
        "structure": torch.zeros(3, 5, 5),
        "furniture": torch.zeros(3, 5, 5),
        "support": torch.zeros(3, 5, 5),
    }
    controls["room_logits"][:, 0, :] = 2.0
    controls["room_logits"][:, 1:, :] = 0.0
    mask = np.ones((5, 5), dtype=bool)
    mask[1:, :] = False
    masks = {
        f"valid_{key}": mask
        for key in ("room_logits", "structure", "furniture", "support")
    }
    controls["structure"][:, :, 0] = 1.0
    controls["furniture"][:, :, 0] = 1.0
    controls["support"][:, :, 0] = 1.0
    filled = inpaint_controls(controls, masks)
    assert float(filled["room_logits"][0, 4, 4]) == 2.0
    assert torch.equal(filled["program"], controls["program"])


def test_nearest_fill_batch_preserves_sample_values() -> None:
    values = torch.zeros(2, 2, 5, 5)
    values[0, :, 0, :] = 2.0
    values[1, :, 0, :] = 3.0
    valid = torch.zeros(2, 5, 5, dtype=torch.bool)
    valid[:, 0, :] = True
    filled = nearest_fill_batch(values, valid)
    assert torch.equal(filled[:, :, 0, :], values[:, :, 0, :])
    assert float(filled[0, 0, 4, 4]) == 2.0
    assert float(filled[1, 0, 4, 4]) == 3.0
