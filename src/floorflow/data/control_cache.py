"""Versioned cache for contour-predicted stage-1 controls."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def control_cache_path(root: str | Path, split: str, name: str) -> Path:
    return Path(root) / split / f"{name.replace('/', '__')}.npz"


def resize_control(value: torch.Tensor, size: int) -> torch.Tensor:
    if value.shape[-2:] == (size, size):
        return value
    return F.interpolate(
        value,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    )


def _float16_control(
    value: torch.Tensor,
    *,
    minimum: float = -65504.0,
    maximum: float = 65504.0,
) -> np.ndarray:
    return (
        torch.nan_to_num(
            value.float(),
            nan=0.0,
            posinf=maximum,
            neginf=minimum,
        )
        .clamp(minimum, maximum)
        .cpu()
        .numpy()
        .astype(np.float16)
    )


def write_control_cache(
    path: Path,
    *,
    room_logits: torch.Tensor,
    program: torch.Tensor,
    structure: torch.Tensor,
    furniture: torch.Tensor,
    support: torch.Tensor,
    room_size: int = 64,
    validity_masks: dict[str, np.ndarray] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "room_logits": _float16_control(
            resize_control(room_logits, room_size)[0],
            minimum=-20.0,
            maximum=20.0,
        ),
        "program": program[0].bool().cpu().numpy(),
        "structure": _float16_control(
            structure[0],
            minimum=0.0,
            maximum=1.0,
        ),
        "furniture": _float16_control(
            furniture[0],
            minimum=0.0,
            maximum=1.0,
        ),
        "support": _float16_control(
            support[0],
            minimum=0.0,
            maximum=1.0,
        ),
    }
    if validity_masks is not None:
        arrays.update(
            {
                key: np.asarray(value, dtype=bool)
                for key, value in validity_masks.items()
            }
        )
    temporary = path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_control_cache(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as payload:
        return {
            key: torch.from_numpy(payload[key].astype(np.float32))
            for key in (
                "room_logits",
                "structure",
                "furniture",
                "support",
            )
        } | {
            "program": torch.from_numpy(payload["program"].astype(bool))
        }


def read_control_validity(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        keys = [key for key in payload.files if key.startswith("valid_")]
        if not keys:
            raise ValueError(f"{path} has no serialized validity masks")
        return {
            key: payload[key].astype(bool)
            for key in keys
        }


class CachedControlDataset(Dataset):
    """Attach cached stage-1 controls to a base floorplan dataset."""

    def __init__(self, base: Dataset, cache_root: str | Path, split: str) -> None:
        self.base = base
        self.cache_root = Path(cache_root)
        self.split = split
        self.taxonomy = base.taxonomy

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        batch = self.base[index]
        _target, _footprint, _floor_id, name = batch[:4]
        controls = read_control_cache(
            control_cache_path(self.cache_root, self.split, str(name))
        )
        return (*batch, controls)


class CachedValidityDataset(CachedControlDataset):
    """Attach controls and serialized validity masks to a base dataset."""

    def __getitem__(self, index: int):
        batch = self.base[index]
        _target, _footprint, _floor_id, name = batch[:4]
        path = control_cache_path(self.cache_root, self.split, str(name))
        return (
            *batch,
            read_control_cache(path),
            read_control_validity(path),
        )


__all__ = [
    "CachedControlDataset",
    "CachedValidityDataset",
    "control_cache_path",
    "read_control_cache",
    "read_control_validity",
    "write_control_cache",
]
