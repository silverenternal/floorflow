from __future__ import annotations

import torch

from floorflow.training.typed_channel_validation import (
    _poisson_binomial_count_probability,
)


def test_poisson_binomial_count_distribution_is_normalized() -> None:
    logits = torch.tensor(
        [[-2.0, 0.0, 2.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    distribution = _poisson_binomial_count_probability(logits)
    assert distribution.shape == (2, 4)
    assert torch.all(distribution >= 0.0)
    assert torch.allclose(distribution.sum(dim=-1), torch.ones(2))
