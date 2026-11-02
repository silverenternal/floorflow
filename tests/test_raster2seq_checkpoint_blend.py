"""Tests for Raster2Seq checkpoint blending edge cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "blend_raster2seq_checkpoints_v1",
    ROOT / "scripts/blend_raster2seq_checkpoints_v1.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("could not load checkpoint blending script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_causal_mask_is_copied_without_nan_at_endpoints():
    mask = torch.triu(torch.ones(4, 4), diagonal=1).masked_fill(
        torch.triu(torch.ones(4, 4), diagonal=1).bool(),
        float("-inf"),
    )
    blended = MODULE.blend_state(
        {"attention_mask": mask, "weight": torch.zeros(1)},
        {"attention_mask": mask, "weight": torch.ones(1)},
        alpha=1.0,
    )

    assert torch.equal(blended["attention_mask"], mask)
    assert not torch.isnan(blended["attention_mask"]).any()
    assert torch.equal(blended["weight"], torch.ones(1))
