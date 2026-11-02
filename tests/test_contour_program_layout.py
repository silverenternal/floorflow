from __future__ import annotations

import torch

from floorflow.heads.contour_program_layout import (
    ContourJointLayoutPredictor,
    ContourProgramLayoutPredictor,
)


def test_contour_program_layout_shapes() -> None:
    model = ContourProgramLayoutPredictor(24, base_channels=8)
    logits = model(
        torch.ones(2, 64, 64),
        torch.zeros(2, 24),
    )
    assert logits.shape == (2, 24, 64, 64)


def test_joint_predictor_requires_only_contour() -> None:
    model = ContourJointLayoutPredictor(12, base_channels=8)
    output = model(torch.ones(2, 1, 64, 64))
    assert output["presence_logits"].shape == (2, 12)
    assert output["layout_logits"].shape == (2, 12, 64, 64)
    output["presence_logits"].mean().backward()
    assert model.program_head[0].weight.grad is not None
