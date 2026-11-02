"""Dataset adapter for cached predicted non-room control rasters."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import (
    CubiCasaFloorplanDataset,
    load_or_rasterize,
    room_instance_targets,
)
from floorflow.data.raster_control import (
    crop_floor_masks,
    fit_channel_raster,
    footprint_center,
    footprint_crop_box,
    load_cubicasa_floor_raster,
    load_cubicasa_raster,
)
from floorflow.data.svg import list_cubicasa_floors


def predicted_control_raster_path(
    root: str | Path,
    split: str,
    name: str,
) -> Path:
    return Path(root) / split / f"{name.replace('/', '__')}.npz"


def predicted_furniture_probability(
    furniture: torch.Tensor,
    room_count: int,
) -> torch.Tensor:
    """Return furniture foreground probability without room/footprint channels."""

    if furniture.ndim != 4:
        raise ValueError("furniture control must be [B,C,H,W]")
    if furniture.shape[1] <= int(room_count):
        raise ValueError("furniture control contains no furniture channels")
    return furniture[:, int(room_count) : -1].amax(dim=1)


class PredictedControlRasterDataset(CubiCasaFloorplanDataset):
    """Attach a target-free structural raster produced from predicted controls."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        control_root: str | Path,
        *,
        image_size: int = 256,
        cache_root: str | Path | None = None,
        augment: bool = False,
        limit: int | None = None,
        write_cache: bool = True,
        page_floor_crop: bool = False,
        include_room_targets: bool = False,
        room_target_limit: int = 24,
    ) -> None:
        super().__init__(
            data_root,
            split,
            image_size=image_size,
            cache_root=cache_root,
            augment=False,
            limit=limit,
            write_cache=write_cache,
        )
        self.control_root = Path(control_root)
        self.control_split = split
        self.control_augment = bool(augment)
        self.page_floor_crop = bool(page_floor_crop)
        self.include_room_targets = bool(include_room_targets)
        self.room_target_limit = int(room_target_limit)

    def _control(self, name: str) -> np.ndarray:
        path = predicted_control_raster_path(
            self.control_root,
            self.control_split,
            name,
        )
        with np.load(path) as payload:
            control = payload["image"].astype(np.float32)
        expected = (3, self.image_size, self.image_size)
        if control.shape != expected:
            raise ValueError(
                f"{path} has control shape {control.shape}, expected {expected}"
            )
        return control

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        target, footprint = load_or_rasterize(
            sample,
            image_size=self.image_size,
            taxonomy=self.taxonomy,
            write_cache=self.write_cache,
        )
        name = f"{sample.relative_case}__{sample.floor_id}"
        control = self._control(name)
        if self.page_floor_crop:
            box = footprint_crop_box(footprint)
            target, footprint = crop_floor_masks(
                target,
                footprint,
                image_size=self.image_size,
            )
            control = control[
                :,
                box[1] : box[3],
                box[0] : box[2],
            ]
            control = fit_channel_raster(
                control,
                image_size=self.image_size,
                fill=1.0,
            )
        if self.control_augment:
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
                control = np.flip(control, axis=2)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
                control = np.flip(control, axis=1)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
                control = np.rot90(control, k=rotations, axes=(1, 2))
        result = (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            torch.from_numpy(control.copy()).float(),
            int(sample.floor_id.rsplit("-", 1)[-1]),
            name,
        )
        if not self.include_room_targets:
            return result
        return (
            *result,
            *room_instance_targets(
                target,
                self.taxonomy,
                max_rooms=self.room_target_limit,
            ),
        )


class DualControlRasterDataset(PredictedControlRasterDataset):
    """Return native RGB and predicted structural control as six channels."""

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        target, footprint = load_or_rasterize(
            sample,
            image_size=self.image_size,
            taxonomy=self.taxonomy,
            write_cache=self.write_cache,
        )
        name = f"{sample.relative_case}__{sample.floor_id}"
        predicted = self._control(name)
        if self.page_floor_crop:
            box = footprint_crop_box(footprint)
            center = footprint_center(footprint)
            target, footprint = crop_floor_masks(
                target,
                footprint,
                image_size=self.image_size,
            )
            predicted = predicted[:, box[1] : box[3], box[0] : box[2]]
            predicted = fit_channel_raster(
                predicted,
                image_size=self.image_size,
                fill=1.0,
            )
            floor_ids = list_cubicasa_floors(sample.svg_path)
            native = load_cubicasa_floor_raster(
                sample.raster_path,
                floor_index=floor_ids.index(str(sample.floor_id)),
                floor_count=len(floor_ids),
                target_center=center,
                image_size=self.image_size,
            )
        else:
            native = load_cubicasa_raster(
                sample.raster_path,
                image_size=self.image_size,
            )
        control = np.concatenate([native, predicted], axis=0)
        if self.control_augment:
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
                control = np.flip(control, axis=2)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
                control = np.flip(control, axis=1)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
                control = np.rot90(control, k=rotations, axes=(1, 2))
        result = (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            torch.from_numpy(control.copy()).float(),
            int(sample.floor_id.rsplit("-", 1)[-1]),
            name,
        )
        if not self.include_room_targets:
            return result
        return (
            *result,
            *room_instance_targets(
                target,
                self.taxonomy,
                max_rooms=self.room_target_limit,
            ),
        )


__all__ = [
    "DualControlRasterDataset",
    "PredictedControlRasterDataset",
    "predicted_control_raster_path",
    "predicted_furniture_probability",
]
