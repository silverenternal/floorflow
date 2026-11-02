"""Source geometry needed by candidate-consensus export must survive scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_chip_official import case_payload


def test_raw_box_export_includes_source_geometry() -> None:
    batch = {
        "_indices": torch.tensor([[7]]),
        "n": torch.tensor(2),
        "boxes": torch.tensor([[[0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 1.0, 2.0]]]),
        "areas": torch.tensor([[2.0, 2.0]]),
        "constraints": torch.zeros(1, 2, 5),
        "b2b": torch.tensor([[[0.0, 1.0, 0.5]]]),
        "p2b": torch.tensor([[[0.0, 0.0, 0.25]]]),
        "pins": torch.tensor([[[0.0, 0.0]]]),
        "n_b2b": torch.tensor([1]),
        "n_p2b": torch.tensor([1]),
        "n_pins": torch.tensor([1]),
    }
    payload = case_payload(
        batch,
        0,
        batch["boxes"][0],
        0.0,
        legalize=False,
        return_raw_boxes=True,
    )
    for key in ("gt_boxes", "areas", "b2b", "p2b", "pins"):
        assert key in payload
    assert payload["index"] == 7
