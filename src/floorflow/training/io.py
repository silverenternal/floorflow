"""Checkpoint, memory-index, and report helpers for room training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.representation.contour_memory import ContourPartitionMemory


def checkpoint_has_module(path: Path | None, prefix: str) -> bool:
    if path is None:
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("head", payload.get("model", payload))
    return any(key.startswith(prefix) for key in state)


def memory_index_by_name(
    memory: ContourPartitionMemory,
    cache_root: Path,
) -> dict[str, int]:
    root = cache_root / "train"
    output = {}
    for index, path in enumerate(memory.paths):
        relative = path.relative_to(root)
        floor_id = path.stem.replace("_", "-")
        output[f"{relative.parent}__{floor_id}"] = index
    return output


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
