"""Adapters for FloorSet-Lite training shards.

The official ``LiteTensorData_v2`` shard has seven batched columns::

    raw, b2b_edges, p2b_edges, pins, tree, fp_sol, metrics

``raw`` stores area plus the five native constraints. An older ten-column
release used dense adjacency matrices and split constraint fields; that format
is accepted as a fallback so previously downloaded caches remain readable.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from floorflow.chip_flow.model import block_features


def load_floorset_columns(path: str | Path) -> tuple[torch.Tensor, ...]:
    """Load and unwrap one supported FloorSet training shard."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, (tuple, list)):
        raise TypeError(f"{path} is not a supported FloorSet shard")
    columns = tuple(payload)
    if len(columns) == 7:
        return columns
    if len(columns) >= 10:
        return columns[:10]
    raise ValueError(f"{path} has {len(columns)} columns; expected 7 or 10")


def is_current_shard(columns: tuple[torch.Tensor, ...]) -> bool:
    """Return whether the shard uses the official seven-column layout."""

    return len(columns) == 7 and columns[0].ndim == 3


def shard_block_count(columns: tuple[torch.Tensor, ...]) -> int:
    """Return the block count shared by all layouts in a training shard."""

    source = columns[0] if is_current_shard(columns) else columns[1]
    if source.ndim != 3:
        raise ValueError("shard layout tensors must have three dimensions")
    return int(source.shape[1])


def edge_codes_to_boundary(edge_codes: torch.Tensor) -> torch.Tensor:
    """Convert FloorSet edge IDs to the five-column boundary bitmask.

    The first code is ``top=1 / bottom=2`` and the second is
    ``left=1 / right=2``. Physics uses ``left=1, right=2, top=4, bottom=8``.
    """

    if edge_codes.ndim != 2 or edge_codes.shape[1] != 2:
        raise ValueError("edge constraints must have shape (blocks, 2)")
    vertical, horizontal = edge_codes.unbind(dim=1)
    return (
        (vertical == 1).to(dtype=torch.float32) * 4.0
        + (vertical == 2).to(dtype=torch.float32) * 8.0
        + (horizontal == 1).to(dtype=torch.float32) * 1.0
        + (horizontal == 2).to(dtype=torch.float32) * 2.0
    )


def adjacency_edges(
    adjacency: torch.Tensor,
    *,
    transposed: bool = False,
) -> torch.Tensor:
    """Convert an adjacency matrix to Float ``(source, target, weight)`` edges."""

    matrix = adjacency.T if transposed else adjacency
    source, target = torch.nonzero(matrix, as_tuple=True)
    weights = matrix[source, target].to(dtype=torch.float32)
    return torch.stack(
        [source.float(), target.float(), weights],
        dim=1,
    )


def b2b_edges(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert an undirected block adjacency matrix to unique weighted edges."""

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("b2b adjacency must be square")
    upper = torch.triu(adjacency.to(dtype=torch.bool), diagonal=1)
    return adjacency_edges(upper)


def p2b_edges(adjacency: torch.Tensor) -> torch.Tensor:
    """Convert ``(block, pin)`` adjacency to ``(pin, block, weight)`` edges."""

    if adjacency.ndim != 2:
        raise ValueError("p2b adjacency must be a matrix")
    return adjacency_edges(adjacency, transposed=True)


def _per_block(value: torch.Tensor, count: int) -> torch.Tensor:
    flat = value.reshape(-1).to(dtype=torch.float32)
    if flat.numel() == 1:
        return flat.expand(count)
    if flat.numel() != count:
        raise ValueError(f"expected one value or {count}, got {flat.numel()}")
    return flat


def case_from_legacy(
    columns: tuple[torch.Tensor, ...],
    layout_index: int,
    *,
    source: str | None = None,
) -> dict:
    """Build one Flow/Physics case from a legacy training shard."""

    layout = tuple(column[layout_index] for column in columns)
    sizes_pos = layout[1].to(dtype=torch.float32)
    if sizes_pos.ndim != 2 or sizes_pos.shape[1] != 4:
        raise ValueError("block_sizes_pos item must be (blocks, 4)")

    width, height, x, y = sizes_pos.unbind(dim=1)
    if torch.any(width <= 0) or torch.any(height <= 0):
        raise ValueError("blocks must have positive width and height")
    count = width.numel()
    boxes = torch.stack([x, y, width, height], dim=1)
    areas = width * height

    pins = layout[2].to(dtype=torch.float32)
    b2b = b2b_edges(layout[3])
    p2b = p2b_edges(layout[4])
    boundary = edge_codes_to_boundary(layout[5])
    preplaced = _per_block(layout[6], count)
    fixed = _per_block(layout[7], count)
    mib = _per_block(layout[8], count)
    cluster = _per_block(layout[9], count)
    constraints = torch.stack(
        [fixed, preplaced, mib, cluster, boundary],
        dim=1,
    )
    features = block_features(areas, constraints, pins, b2b)
    return {
        "areas": areas,
        "boxes": boxes,
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "constraints": constraints,
        "features": features,
        "source": source,
        "source_index": int(layout_index),
    }


def case_from_current(
    columns: tuple[torch.Tensor, ...],
    layout_index: int,
    *,
    source: str | None = None,
) -> dict:
    """Build one case from the official seven-column shard."""

    raw, b2b, p2b, pins, _tree, fp_sol, _metrics = (
        column[layout_index] for column in columns
    )
    raw = raw.to(dtype=torch.float32)
    valid = raw[:, 0] > 0
    valid_rows = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_rows.numel() and not torch.equal(
        valid_rows,
        torch.arange(valid_rows.numel(), device=valid_rows.device),
    ):
        raise ValueError(
            "current FloorSet layouts must contain a contiguous valid block prefix"
        )
    raw = raw[valid]
    fp_sol = fp_sol[: raw.shape[0]].to(dtype=torch.float32)

    width, height, x, y = fp_sol.unbind(dim=1)
    boxes = torch.stack([x, y, width, height], dim=1)
    areas = raw[:, 0]
    constraints = raw[:, 1:6].clone()
    shapes = [
        (round(width_value), round(height_value))
        for width_value, height_value in zip(width.tolist(), height.tolist())
    ]
    counts = Counter(shapes)
    top = max(counts.values())
    constraints[:, 2] = 0.0
    if top > 1:
        shape = min(candidate for candidate, count in counts.items() if count == top)
        constraints[:, 2] = torch.tensor(
            [float(item == shape) for item in shapes],
            dtype=constraints.dtype,
        )
    pins = pins.to(dtype=torch.float32)
    b2b = b2b.to(dtype=torch.float32)
    p2b = p2b.to(dtype=torch.float32)
    features = block_features(areas, constraints, pins, b2b)
    return {
        "areas": areas,
        "boxes": boxes,
        "b2b": b2b,
        "p2b": p2b,
        "pins": pins,
        "constraints": constraints,
        "features": features,
        "source": source,
        "source_index": int(layout_index),
    }


def case_from_shard(
    columns: tuple[torch.Tensor, ...],
    layout_index: int,
    *,
    source: str | None = None,
) -> dict:
    """Dispatch between supported FloorSet shard layouts."""

    builder = case_from_current if is_current_shard(columns) else case_from_legacy
    return builder(columns, layout_index, source=source)


__all__ = [
    "adjacency_edges",
    "b2b_edges",
    "case_from_current",
    "case_from_legacy",
    "case_from_shard",
    "edge_codes_to_boundary",
    "is_current_shard",
    "load_floorset_columns",
    "p2b_edges",
    "shard_block_count",
]
