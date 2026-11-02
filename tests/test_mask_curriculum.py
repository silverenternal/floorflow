from __future__ import annotations

import torch

from floorflow.training.mask_curriculum import MaskRatioCurriculum


def test_mask_curriculum_contains_fixed_anchors_and_bounds() -> None:
    curriculum = MaskRatioCurriculum()
    values = [
        curriculum.sample(progress=index / 100)
        for index in range(200)
    ]
    assert 1.0 in values
    assert 0.0 in values
    assert 0.25 in values
    assert all(0.0 <= value <= 1.0 for value in values)


def test_mask_curriculum_accepts_device() -> None:
    value = MaskRatioCurriculum().sample(0.5, device=torch.device("cpu"))
    assert isinstance(value, float)
