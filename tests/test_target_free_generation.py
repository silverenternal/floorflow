from __future__ import annotations

import pytest
import torch

from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.target_free import (
    GenerationRequest,
    TargetLeakError,
    forbidden_parameters,
    target_free_generation,
)


def test_production_overlay_decoder_exposes_no_target_input() -> None:
    assert forbidden_parameters(decode_overlay) == ()
    assert getattr(decode_overlay, "target_free", False) is True


def _leaky_target(footprint, target):
    return footprint


def _leaky_ground_truth(footprint, ground_truth):
    return footprint


def _leaky_gt(footprint, gt_labels):
    return footprint


def _leaky_oracle(footprint, oracle_controls):
    return footprint


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (_leaky_target, "target"),
        (_leaky_ground_truth, "ground_truth"),
        (_leaky_gt, "gt_labels"),
        (_leaky_oracle, "oracle_controls"),
    ],
)
def test_target_free_decorator_rejects_tainted_signatures(
    function,
    parameter: str,
) -> None:
    with pytest.raises(TargetLeakError, match=parameter):
        target_free_generation(function)


def test_generation_request_has_only_footprint_and_explicit_metadata() -> None:
    request = GenerationRequest(
        footprint=torch.ones(2, 1, 8, 8),
        floor_ids=torch.zeros(2, dtype=torch.long),
        style_ids=torch.ones(2, dtype=torch.long),
    )
    assert set(request.as_kwargs()) == {"footprint", "floor_ids", "style_ids"}
