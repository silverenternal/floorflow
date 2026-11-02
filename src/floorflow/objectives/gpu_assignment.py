"""GPU-friendly soft bipartite matching for training-time alignment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def exact_assignment_batch(
    costs: Sequence[torch.Tensor],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solve exact assignments, batching the host copy when shapes align."""

    if not costs:
        return []
    if any(cost.ndim != 2 for cost in costs):
        raise ValueError("each assignment cost must be a matrix")
    device = costs[0].device
    if any(cost.device != device for cost in costs):
        raise ValueError("assignment costs must share a device")
    rows = [int(cost.shape[0]) for cost in costs]
    columns = [int(cost.shape[1]) for cost in costs]
    assignments: list[tuple[np.ndarray, np.ndarray]] = []
    if not any(rows) or not any(columns):
        return [
            (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
            for _ in costs
        ]
    if len(set(zip(rows, columns))) == 1:
        host_costs = torch.stack(
            [cost.detach() for cost in costs]
        ).float().cpu().numpy()
    else:
        host_costs = [
            cost.detach().float().cpu().numpy()
            if cost.numel()
            else np.zeros((0, 0), dtype=np.float32)
            for cost in costs
        ]
    for index, (row_count, column_count) in enumerate(zip(rows, columns)):
        if row_count == 0 or column_count == 0:
            assignments.append(
                (
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                )
            )
            continue
        host_cost = (
            host_costs[index]
            if isinstance(host_costs, list)
            else host_costs[index, :row_count, :column_count]
        )
        predicted, target = linear_sum_assignment(
            host_cost
        )
        assignments.append(
            (
                predicted.astype(np.int64, copy=False),
                target.astype(np.int64, copy=False),
            )
        )
    return assignments


def assignment_index_tensors(
    assignments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack variable-length assignments and transfer all indices once."""

    batch_size = len(assignments)
    max_matches = max(
        (len(predicted) for predicted, _target in assignments),
        default=0,
    )
    packed = np.zeros((batch_size, 3, max_matches), dtype=np.int64)
    for batch, (predicted, target) in enumerate(assignments):
        count = min(len(predicted), len(target))
        packed[batch, 0, :count] = predicted[:count]
        packed[batch, 1, :count] = target[:count]
        packed[batch, 2, :count] = 1
    tensor = torch.from_numpy(packed).to(device)
    return tensor[:, 0], tensor[:, 1], tensor[:, 2].bool()


def padded_exact_assignment_batch(
    costs: Sequence[torch.Tensor],
    *,
    padding_cost: float = 1.0e6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solve ragged rectangular assignments after one packed host transfer."""

    if not costs:
        return []
    if any(cost.ndim != 2 for cost in costs):
        raise ValueError("each assignment cost must be a matrix")
    device = costs[0].device
    if any(cost.device != device for cost in costs):
        raise ValueError("assignment costs must share a device")
    rows = [int(cost.shape[0]) for cost in costs]
    columns = [int(cost.shape[1]) for cost in costs]
    size = max(max(rows, default=0), max(columns, default=0), 1)
    packed = torch.full(
        (len(costs), size, size),
        float(padding_cost),
        dtype=torch.float32,
        device=device,
    )
    for index, cost in enumerate(costs):
        if cost.numel():
            packed[index, : rows[index], : columns[index]] = cost.float()
    host = packed.detach().cpu().numpy()
    assignments: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (row_count, column_count) in enumerate(zip(rows, columns)):
        selected_rows, selected_columns = linear_sum_assignment(host[index])
        keep = (selected_rows < row_count) & (selected_columns < column_count)
        assignments.append(
            (
                selected_rows[keep].astype(np.int64, copy=False),
                selected_columns[keep].astype(np.int64, copy=False),
            )
        )
    return assignments


def sinkhorn_log_assignment(
    cost: torch.Tensor,
    *,
    row_valid: torch.Tensor | None = None,
    column_valid: torch.Tensor | None = None,
    iterations: int = 10,
    temperature: float = 0.1,
) -> torch.Tensor:
    if cost.ndim != 3:
        raise ValueError("cost must be [B,rows,columns]")
    logits = -cost / max(float(temperature), 1.0e-6)
    if row_valid is None:
        row_valid = torch.ones(cost.shape[:2], dtype=torch.bool, device=cost.device)
    if column_valid is None:
        column_valid = torch.ones(
            cost.shape[0],
            cost.shape[2],
            dtype=torch.bool,
            device=cost.device,
        )
    valid = row_valid[:, :, None] & column_valid[:, None, :]
    logits = logits.masked_fill(~valid, -1.0e4)
    for _ in range(max(1, int(iterations))):
        logits = logits - torch.logsumexp(logits, dim=2, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    assignment = logits.exp() * valid
    return assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1.0e-9)


def soft_room_assignment(
    predicted_boxes: torch.Tensor,
    reference_boxes: torch.Tensor,
    *,
    predicted_valid: torch.Tensor | None = None,
    reference_valid: torch.Tensor | None = None,
    iterations: int = 10,
) -> torch.Tensor:
    centers = predicted_boxes[..., :2] * 0.5 + predicted_boxes[..., 2:] * 0.5
    reference_centers = (
        reference_boxes[..., :2] * 0.5 + reference_boxes[..., 2:] * 0.5
    )
    cost = torch.cdist(centers, reference_centers)
    return sinkhorn_log_assignment(
        cost,
        row_valid=predicted_valid,
        column_valid=reference_valid,
        iterations=iterations,
    )


__all__ = [
    "assignment_index_tensors",
    "exact_assignment_batch",
    "padded_exact_assignment_batch",
    "sinkhorn_log_assignment",
    "soft_room_assignment",
]
