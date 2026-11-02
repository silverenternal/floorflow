"""Shared datasets and control fusion for residential structure training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from floorflow.data.raster_control import load_cubicasa_raster


class PredictedTargetDataset(Dataset):
    def __init__(self, root: Path, split: str, target_root: Path) -> None:
        self.root = root
        self.split = split
        self.target_root = target_root
        self.names = [
            path.stem for path in sorted((root / split).glob("*.npz"))
        ]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        category, case_id, floor = name.split("__", maxsplit=2)
        path = (
            self.target_root
            / self.split
            / category
            / case_id
            / f"{floor.replace('-', '_')}.npz"
        )
        with np.load(path) as payload:
            target = torch.from_numpy(payload["target"].astype(np.int64))
            footprint = torch.from_numpy(payload["footprint"].astype(bool))
        return target, footprint, 0, name


class RasterPredictedControlDataset(PredictedTargetDataset):
    def __init__(
        self,
        predicted_root: Path,
        split: str,
        target_root: Path,
        data_root: Path,
        *,
        image_size: int = 256,
        limit: int | None = None,
        offset: int = 0,
    ) -> None:
        super().__init__(predicted_root, split, target_root)
        self.split = split
        self.data_root = data_root
        self.image_size = int(image_size)
        self.names = self.names[int(offset) :]
        if limit is not None:
            self.names = self.names[: int(limit)]

    def _raster(self, name: str) -> torch.Tensor:
        category, case_id, floor = name.split("__", maxsplit=2)
        floor_number = floor.rsplit("-", 1)[-1]
        path = (
            self.data_root
            / category
            / case_id
            / f"F{floor_number}_scaled.png"
        )
        return torch.from_numpy(
            load_cubicasa_raster(path, image_size=self.image_size)
        ).float()

    def __getitem__(self, index: int):
        target, footprint, floor_ids, name = super().__getitem__(index)
        return target, footprint, floor_ids, name, self._raster(name)


def bridge_controls(
    predicted: dict[str, torch.Tensor],
    raster_room_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    controls = dict(predicted)
    controls["room_logits"] = raster_room_logits
    return controls


__all__ = [
    "PredictedTargetDataset",
    "RasterPredictedControlDataset",
    "bridge_controls",
]
