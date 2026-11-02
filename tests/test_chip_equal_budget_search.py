"""Tests for label-free equal-budget proposal selection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_chip_equal_budget_search import select_proposals


def test_select_proposals_chooses_lowest_label_free_score_per_case() -> None:
    proposals = [
        torch.tensor([[[1.0, 2.0, 0.0]], [[3.0, 4.0, 0.0]]]),
        torch.tensor([[[5.0, 6.0, 0.0]], [[7.0, 8.0, 0.0]]]),
    ]
    scores = torch.tensor([[0.2, 0.9], [0.8, 0.1]])

    selected, indices = select_proposals(proposals, scores)

    assert indices.tolist() == [0, 1]
    assert selected.tolist() == [[[1.0, 2.0, 0.0]], [[7.0, 8.0, 0.0]]]
