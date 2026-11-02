"""Tests for host-wall target alignment and sensitivity."""

import numpy as np
import pytest
import torch

from floorflow.objectives.vector_primitive_loss import (
    aligned_host_targets,
    host_accuracy,
)


def test_aligned_host_targets_follow_wall_matching() -> None:
    logits = torch.zeros(1, 3, 4)
    target_hosts = torch.tensor([[2, 0, 1]])
    wall_pairs = [(np.array([0, 1, 2]), np.array([2, 0, 1]))]
    opening_pairs = [
        (np.array([0, 1, 2]), np.array([0, 1, 2]))
    ]

    aligned = aligned_host_targets(
        logits,
        target_hosts,
        wall_pairs,
        opening_pairs,
    )

    assert aligned.tolist() == [[0, 1, 2]]
    assert host_accuracy(logits, aligned, opening_pairs) == pytest.approx(
        1.0 / 3.0
    )
