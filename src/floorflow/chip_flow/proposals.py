"""Deterministic source layouts and token transport for chip flow matching."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from floorflow.chip_flow.model import encode_latent


def shelf_pack(areas: torch.Tensor) -> torch.Tensor:
    """Return the deterministic shelf-packed source boxes ``(x, y, w, h)``."""

    count = areas.numel()
    width = areas.sum().sqrt() * 1.05
    order = torch.argsort(areas, descending=True)
    boxes = torch.zeros(count, 4, dtype=areas.dtype)
    cursor_x = cursor_y = row_height = 0.0
    for index in order.tolist():
        side = float(areas[index]) ** 0.5
        if cursor_x + side > width and cursor_x > 0:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0
        boxes[index] = torch.tensor(
            [cursor_x, cursor_y, side, side],
            dtype=areas.dtype,
        )
        cursor_x += side
        row_height = max(row_height, side)
    return boxes


def token_ot_reorder(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Reorder target block tokens with the exact minimum squared-cost matching."""

    cost = torch.cdist(source, target).pow(2).detach().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    order = torch.empty(target.shape[0], dtype=torch.long)
    order[torch.as_tensor(rows)] = torch.as_tensor(columns)
    return target[order]


class TokenOTCache:
    """Bounded cache of reordered target tokens keyed by case/source identity."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("OT cache size must be positive")
        self.max_entries = int(max_entries)
        self.entries: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[int, str]) -> torch.Tensor | None:
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return value

    def set(self, key: tuple[int, str], value: torch.Tensor) -> None:
        self.entries[key] = value.detach().clone()
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)


def _source_key(case_index: int, source: torch.Tensor) -> tuple[int, str]:
    digest = hashlib.sha1(
        source.contiguous().numpy().tobytes(),
        usedforsecurity=False,
    ).hexdigest()
    return int(case_index), digest


def token_ot_reorder_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    workers: int = 4,
    case_indices: np.ndarray | None = None,
    cache: TokenOTCache | None = None,
) -> torch.Tensor:
    """Reorder batched target tokens with parallel CPU OT solvers."""

    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("token OT expects matching batched [B, N, D] tensors")
    if case_indices is not None and len(case_indices) != source.shape[0]:
        raise ValueError("case_indices must match the OT batch size")
    reordered = target.clone()
    source_cpu = source.detach().cpu()
    pending = []
    keys = {}
    for row in range(source.shape[0]):
        if cache is not None and case_indices is not None:
            key = _source_key(case_indices[row], source_cpu[row])
            cached = cache.get(key)
            if cached is not None:
                reordered[row] = cached.to(target.device)
                continue
            keys[row] = key
        pending.append(row)
    if not pending:
        return reordered
    target_cpu = target.detach().cpu()[pending]
    costs = torch.cdist(source_cpu[pending], target_cpu).pow(2).numpy()

    def solve(row: int):
        return linear_sum_assignment(costs[row])

    worker_count = max(1, min(int(workers), len(pending)))
    if worker_count == 1:
        assignments = [solve(row) for row in range(len(costs))]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            assignments = list(executor.map(solve, range(len(costs))))
    for local_row, (rows, columns) in enumerate(assignments):
        row = pending[local_row]
        reordered[row, rows] = target[row, columns]
        if cache is not None and row in keys:
            cache.set(keys[row], reordered[row])
    return reordered


def prepare_flow_pair(
    boxes: torch.Tensor,
    areas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the source and target latents without changing token conditions."""

    source = encode_latent(shelf_pack(areas), areas)
    target = encode_latent(boxes, areas)
    return source, target


__all__ = [
    "TokenOTCache",
    "prepare_flow_pair",
    "shelf_pack",
    "token_ot_reorder",
    "token_ot_reorder_batch",
]
