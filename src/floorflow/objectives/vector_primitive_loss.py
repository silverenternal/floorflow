"""Matched geometry and host-wall losses for vector primitives."""

from __future__ import annotations

import numpy as np
import torch


def matched_wall_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    losses = []
    for batch, (rows, columns) in enumerate(pairs):
        if not rows.size:
            continue
        rows_t = torch.from_numpy(rows).to(predicted.device)
        columns_t = torch.from_numpy(columns).to(predicted.device)
        direct = torch.nn.functional.smooth_l1_loss(
            predicted[batch, rows_t],
            target[batch, columns_t],
            reduction="none",
        ).mean(dim=(1, 2))
        reverse = torch.nn.functional.smooth_l1_loss(
            predicted[batch, rows_t],
            target[batch, columns_t][:, [1, 0]],
            reduction="none",
        ).mean(dim=(1, 2))
        losses.append(torch.minimum(direct, reverse).mean())
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.0


def aligned_host_targets(
    logits: torch.Tensor | None,
    target_hosts: torch.Tensor | None,
    wall_pairs: list[tuple[np.ndarray, np.ndarray]],
    opening_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor | None:
    """Map target wall ids to matched predicted wall-query ids."""

    if logits is None or target_hosts is None:
        return None
    aligned = torch.zeros(
        logits.shape[:2],
        dtype=torch.long,
        device=target_hosts.device,
    )
    for batch, pair_group in enumerate(zip(wall_pairs, opening_pairs, strict=True)):
        (wall_rows, wall_columns), (opening_rows, opening_columns) = pair_group
        target_to_predicted = {
            int(column): int(row)
            for row, column in zip(wall_rows, wall_columns, strict=True)
        }
        for predicted_opening, target_opening in zip(
            opening_rows,
            opening_columns,
            strict=True,
        ):
            target_wall = int(target_hosts[batch, target_opening])
            if target_wall in target_to_predicted:
                aligned[batch, int(predicted_opening)] = target_to_predicted[
                    target_wall
                ]
    return aligned


def host_loss(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    reference: torch.Tensor,
) -> torch.Tensor:
    if logits is None or target is None:
        return reference.sum() * 0.0
    losses = []
    for batch, (rows, _columns) in enumerate(pairs):
        if not rows.size:
            continue
        row_ids = torch.from_numpy(rows).to(logits.device)
        losses.append(
            torch.nn.functional.cross_entropy(
                logits[batch, row_ids],
                target[batch, row_ids],
            )
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def host_accuracy(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if logits is None or target is None:
        return 0.0
    correct = 0
    total = 0
    for batch, (rows, _columns) in enumerate(pairs):
        row_ids = torch.from_numpy(rows).to(logits.device)
        correct += int(
            (logits[batch, row_ids].argmax(dim=-1) == target[batch, row_ids])
            .sum()
            .item()
        )
        total += int(rows.size)
    return correct / max(1, total)


__all__ = [
    "aligned_host_targets",
    "host_accuracy",
    "host_loss",
    "matched_wall_loss",
]
