from __future__ import annotations

import torch

from floorflow.inference.dual_pass import (
    confidence_map,
    sample_dual_pass,
    select_remask,
)


class _DummyFlow:
    target_channels = 3

    def forward_with_aux(self, state, time, footprint, **kwargs):
        del time, footprint, kwargs
        return -0.1 * state, {}


def test_confidence_remask_selects_exact_low_confidence_pixels() -> None:
    prediction = torch.zeros(1, 4, 8, 8)
    prediction[:, 0, :4] = 8.0
    prediction[:, 1, 4:] = 8.0
    footprint = torch.ones(1, 1, 8, 8)
    confidence = confidence_map(prediction, footprint)
    remask = select_remask(confidence, footprint, ratio=0.25)
    assert int(remask.sum()) == 16


def test_dual_pass_is_single_candidate_and_records_remask() -> None:
    model = _DummyFlow()
    footprint = torch.ones(1, 1, 8, 8)
    control = torch.zeros(1, 3, 8, 8)
    mask = footprint.expand_as(control)
    result = sample_dual_pass(
        model,
        footprint,
        control,
        mask,
        torch.zeros(3),
        steps_per_pass=1,
        remask_ratio=0.5,
    )
    assert result.final.shape == control.shape
    assert result.draft.shape == control.shape
    assert int(result.remask.sum()) == 32
    assert result.diagnostics["candidate_count"] == 1.0
    assert result.diagnostics["passes"] == 2.0
