"""Opening matching utilities for the unified generator loss."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from floorflow.objectives.gpu_assignment import exact_assignment_batch


def match_openings(
    centers: torch.Tensor,
    widths: torch.Tensor,
    target_centers: torch.Tensor,
    target_widths: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    target_counts: Sequence[int] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], torch.Tensor]:
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )
    if len(counts) != centers.shape[0]:
        raise ValueError("target_counts must match the opening batch")
    costs = []
    populated_batches = []
    for batch in range(centers.shape[0]):
        count = counts[batch]
        if count == 0:
            continue
        cost = torch.cdist(
            centers[batch],
            target_centers[batch, :count],
        ) + 0.5 * (
            widths[batch, :, None] - target_widths[batch, None, :count]
        ).abs()
        costs.append(cost)
        populated_batches.append(batch)
    solved = exact_assignment_batch(costs)
    solved_by_batch = dict(zip(populated_batches, solved))
    expected_host = np.zeros(tuple(widths.shape), dtype=np.float32)
    pairs = []
    for batch in range(centers.shape[0]):
        assignment = solved_by_batch.get(batch)
        if assignment is None:
            empty = np.zeros(0, dtype=np.int64)
            pairs.append((empty, empty))
            continue
        rows, columns = assignment
        expected_host[batch, rows] = 1.0
        pairs.append((rows, columns))
    expected = torch.from_numpy(expected_host).to(
        device=widths.device,
        dtype=widths.dtype,
    )
    return pairs, expected


__all__ = ["match_openings"]
