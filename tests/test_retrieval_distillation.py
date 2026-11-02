from pathlib import Path

import pytest
import torch

from floorflow.training.retrieval_distillation import (
    distillation_loss,
    excluded_indices,
    name_to_cache_path,
)


def test_training_name_maps_to_retrieval_cache_path() -> None:
    assert name_to_cache_path(
        "high_quality_architectural/6044__Floor-1"
    ) == Path("high_quality_architectural/6044/Floor_1.npz")


def test_leave_one_out_indices_are_explicit() -> None:
    mapping = {
        Path("high_quality_architectural/6044/Floor_1.npz"): 7,
    }
    indices = excluded_indices(
        ["high_quality_architectural/6044__Floor-1"],
        mapping,
    )
    assert torch.equal(indices, torch.tensor([7]))


def test_distillation_loss_is_zero_for_identical_teacher() -> None:
    logits = torch.randn(2, 4, 8, 8)
    loss = distillation_loss(logits, logits, temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1.0e-5)
    assert torch.isfinite(loss)
