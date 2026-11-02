from __future__ import annotations

import pytest
import torch

from floorflow.heads.attention_masks import unmask_first_empty_row


def test_unmask_first_empty_row_preserves_one_valid_attention_key() -> None:
    padding = torch.tensor(
        [
            [True, True, True],
            [True, False, True],
            [False, False, True],
        ]
    )
    actual = unmask_first_empty_row(padding)
    assert actual.tolist() == [
        [False, True, True],
        [True, False, True],
        [False, False, True],
    ]
    assert padding[0].all()


def test_unmask_first_empty_row_handles_zero_key_sequences() -> None:
    padding = torch.empty((2, 0), dtype=torch.bool)
    actual = unmask_first_empty_row(padding)
    assert actual.shape == padding.shape
    assert actual.dtype == torch.bool


def test_unmask_first_empty_row_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[B,N\]"):
        unmask_first_empty_row(torch.ones(2, 3, 4, dtype=torch.bool))
