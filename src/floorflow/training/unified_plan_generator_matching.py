"""Matching helpers for unified floorplan-generator supervision."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from floorflow.objectives.gpu_assignment import exact_assignment_batch


def _target_counts(
    target_valid: torch.Tensor,
    target_counts: Sequence[int] | None,
) -> list[int]:
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )
    if len(counts) != target_valid.shape[0]:
        raise ValueError("target_counts must match the target batch")
    return counts


def match_candidate_queries(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    target_counts: Sequence[int] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Assign one query to every target before classifying unmatched queries."""

    counts = _target_counts(target_valid, target_counts)
    costs = []
    populated_batches = []
    selected_rows: list[np.ndarray | None] = []
    for batch in range(predicted.shape[0]):
        count = counts[batch]
        if count == 0:
            continue
        direct = (
            predicted[batch][:, None] - target[batch, :count][None]
        ).abs().mean(dim=(2, 3))
        reverse = (
            predicted[batch][:, None]
            - target[batch, :count][None, :, [1, 0]]
        ).abs().mean(dim=(2, 3))
        pred_length = (predicted[batch, :, 1] - predicted[batch, :, 0]).norm(dim=-1)
        target_length = (
            target[batch, :count, 1] - target[batch, :count, 0]
        ).norm(dim=-1)
        cost = torch.minimum(direct, reverse) + 0.5 * (
            pred_length[:, None] - target_length[None]
        ).abs()
        row_indices = candidate_valid[batch].nonzero(as_tuple=False).flatten()
        if row_indices.numel() == predicted.shape[1]:
            costs.append(cost)
            selected_rows.append(None)
        else:
            costs.append(cost.index_select(0, row_indices))
            selected_rows.append(row_indices.cpu().numpy())
        populated_batches.append(batch)
    solved = exact_assignment_batch(costs)
    solved_by_batch = dict(zip(populated_batches, solved))
    pairs = []
    positive = []
    for batch in range(predicted.shape[0]):
        assignment = solved_by_batch.get(batch)
        if assignment is None:
            pairs.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
            positive.append(np.zeros(predicted.shape[1], dtype=bool))
            continue
        rows, columns = assignment
        population_index = populated_batches.index(batch)
        selected = selected_rows[population_index]
        if selected is not None:
            rows = selected[rows].astype(np.int64, copy=False)
        mask = np.zeros(predicted.shape[1], dtype=bool)
        mask[rows] = True
        pairs.append((rows, columns))
        positive.append(mask)
    return pairs, positive


def balanced_positive_weight(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Balance sparse positive query masks without exceeding a stable cap."""

    positive = target.sum()
    negative = target.numel() - positive
    return (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0).to(logits.dtype)


__all__ = ["balanced_positive_weight", "match_candidate_queries"]
