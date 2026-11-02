"""Tests for room-mask induced partition edge evidence."""

import torch

from floorflow.geometry.partition_edge_evidence import partition_edge_evidence


def test_partition_edge_is_high_between_adjacent_room_supports() -> None:
    logits = torch.full((1, 2, 16, 16), -10.0)
    logits[0, 0, :, :8] = 10.0
    logits[0, 1, :, 8:] = 10.0

    evidence = partition_edge_evidence(logits)

    assert evidence.shape == (1, 16, 16)
    assert evidence[0, :, 7].mean() > 0.9
    assert evidence[0, :, 2:6].mean() < 0.05
