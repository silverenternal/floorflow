"""Vector-primitive targets paired with native CubiCasa rasters."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.data.raster_control import load_cubicasa_raster
from floorflow.data.vector_geometry_dataset import VectorGeometryDataset


class RasterVectorGeometryDataset(VectorGeometryDataset):
    def __init__(
        self,
        geometry_root: str | Path,
        data_root: str | Path,
        split: str,
        *,
        offset: int = 0,
        **kwargs,
    ) -> None:
        limit = kwargs.pop("limit", None)
        super().__init__(geometry_root, split, limit=None, **kwargs)
        self.paths = self.paths[int(offset) :]
        if limit is not None:
            self.paths = self.paths[: int(limit)]
        self.data_root = Path(data_root)
        self.split = split

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        path = self.paths[index]
        relative = path.relative_to(self.geometry_root / self.split)
        floor_name = path.stem.removesuffix(".geometry")
        floor_number = floor_name.rsplit("_", 1)[-1]
        image_path = (
            self.data_root
            / relative.parent
            / f"F{floor_number}_scaled.png"
        )
        item["image"] = torch.from_numpy(
            load_cubicasa_raster(
                image_path,
                image_size=self.image_size,
            )
        ).float()
        return item


__all__ = ["RasterVectorGeometryDataset"]
