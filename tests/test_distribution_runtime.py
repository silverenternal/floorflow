import pytest
import torch

from floorflow.training.distribution_runtime import apply_style_dropout


def test_style_dropout_is_batch_optional_without_shape_corruption() -> None:
    style_ids = torch.tensor([0, 2, 5])
    unchanged = apply_style_dropout(style_ids, 0.0)
    assert unchanged is style_ids
    assert apply_style_dropout(style_ids, 1.0 - 1.0e-8) is None


def test_style_dropout_rejects_invalid_probability() -> None:
    style_ids = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="dropout"):
        apply_style_dropout(style_ids, -0.1)
