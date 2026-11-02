"""Tests for raw-to-legalized candidate ordering retention."""

from __future__ import annotations

import torch

from floorflow.chip_flow.legalization_survival import (
    legalization_rank_retention_loss,
    legalization_survival,
)


def _view(raw: float, legalized: float) -> dict:
    return {
        "stages": {
            "refined": {"b2b_hpwl_gap": raw},
            "legalized": {"b2b_hpwl_gap": legalized},
        }
    }


def test_legalization_survival_separates_preserved_and_reversed_pairs() -> None:
    rows = [
        {
            "index": 0,
            "baseline": _view(0.2, 0.2),
            "expanded": _view(0.1, 0.1),
            "projector32": _view(0.3, 0.05),
        }
    ]
    report = legalization_survival(rows)
    b2b = report["metrics"]["b2b_hpwl_gap"]
    assert b2b["defined"] == 3
    assert b2b["preserved"] == 1
    assert b2b["reversed"] == 2
    assert b2b["top_choice_preserved"] == 0


def test_legalization_survival_fails_closed_without_pairs() -> None:
    report = legalization_survival(
        [{"index": 0, "baseline": _view(0.2, 0.1)}]
    )
    assert report["status"] == "insufficient_data"
    assert report["summary"]["pairwise_lsr"] is None


def test_legalization_rank_retention_loss_is_differentiable() -> None:
    predicted = torch.tensor([[0.2, 0.1, 0.4]], requires_grad=True)
    legalized = torch.tensor([[0.3, 0.2, 0.1]])
    loss = legalization_rank_retention_loss(predicted, legalized)
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0
