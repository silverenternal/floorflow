from __future__ import annotations

import torch

from floorflow.training.crop_curriculum import random_crop_and_resize


def test_random_crop_and_resize_preserves_shapes_and_dtype() -> None:
    target = torch.arange(2 * 32 * 32).reshape(2, 32, 32) % 7
    footprint = torch.zeros_like(target)
    footprint[:, 4:28, 4:28] = 1
    generator = torch.Generator().manual_seed(7)
    cropped_target, cropped_footprint = random_crop_and_resize(
        target,
        footprint,
        probability=1.0,
        minimum_fraction=0.5,
        generator=generator,
    )
    assert cropped_target.shape == target.shape
    assert cropped_footprint.shape == footprint.shape
    assert cropped_target.dtype == target.dtype
    assert set(torch.unique(cropped_footprint).tolist()) <= {0, 1}
