"""Stack cached controls for deterministic completion evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.control_cache import control_cache_path, read_control_cache
from floorflow.inference.contour_controls import ContourControls


def load_cached_contour_controls(
    root: str | Path,
    split: str,
    names: tuple[str, ...] | list[str],
    device: torch.device,
) -> ContourControls:
    rows = [
        read_control_cache(control_cache_path(root, split, str(name)))
        for name in names
    ]
    return ContourControls(
        room_logits=torch.stack([row["room_logits"] for row in rows]).to(device),
        program=torch.stack([row["program"] for row in rows]).to(device),
        structure=torch.stack([row["structure"] for row in rows]).to(device),
        furniture=torch.stack([row["furniture"] for row in rows]).to(device),
        support=torch.stack([row["support"] for row in rows]).to(device),
        auxiliary={},
    )


__all__ = ["load_cached_contour_controls"]
