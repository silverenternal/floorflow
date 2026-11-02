"""CubiCasa5K split handling and cached semantic-floorplan dataset."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from floorflow.data.svg import list_cubicasa_floors, rasterize_cubicasa_svg
from floorflow.data.taxonomy import FloorplanTaxonomy, build_taxonomy, scan_taxonomy_counts


@dataclass(frozen=True)
class FloorplanSample:
    relative_case: Path
    svg_path: Path
    cache_path: Path | None
    floor_id: str = "Floor-1"

    @property
    def raster_path(self) -> Path:
        floor_number = self.floor_id.rsplit("-", 1)[-1]
        return self.svg_path.parent / f"F{floor_number}_scaled.png"


def read_split(split_file: str | Path) -> list[Path]:
    split_file = Path(split_file)
    cases: list[Path] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        value = line.strip().strip("/")
        if value:
            cases.append(Path(value))
    return cases


def collect_samples(
    data_root: str | Path,
    split: str,
    *,
    cache_root: str | Path | None = None,
    limit: int | None = None,
    floors: bool = True,
) -> list[FloorplanSample]:
    data_root = Path(data_root)
    cache_root = Path(cache_root) if cache_root else None
    samples: list[FloorplanSample] = []
    for relative_case in read_split(data_root / f"{split}.txt"):
        svg_path = data_root / relative_case / "model.svg"
        if not svg_path.is_file():
            continue
        floor_ids = list_cubicasa_floors(svg_path) if floors else ["Floor-1"]
        for floor_id in floor_ids:
            cache_path = (
                cache_root / split / relative_case / f"{floor_id.replace('-', '_')}.npz"
                if cache_root
                else None
            )
            samples.append(FloorplanSample(relative_case, svg_path, cache_path, floor_id))
            if limit is not None and len(samples) >= int(limit):
                return samples
    return samples


def load_or_rasterize(
    sample: FloorplanSample,
    *,
    image_size: int,
    taxonomy: FloorplanTaxonomy | None = None,
    write_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if sample.cache_path is not None and sample.cache_path.is_file():
        with np.load(sample.cache_path) as payload:
            target = payload["target"].astype(np.uint8, copy=False)
            footprint = payload["footprint"].astype(np.uint8, copy=False)
            if target.shape == (image_size, image_size) and footprint.shape == (image_size, image_size):
                return target, footprint
            if write_cache:
                raise ValueError(
                    f"cache resolution mismatch at {sample.cache_path}: "
                    f"found {target.shape}, expected {(image_size, image_size)}; "
                    "use a separate cache root or rebuild explicitly"
                )

    target, footprint = rasterize_cubicasa_svg(
        sample.svg_path,
        image_size=image_size,
        floor_id=sample.floor_id,
        taxonomy=taxonomy,
    )
    if write_cache and sample.cache_path is not None:
        sample.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = sample.cache_path.with_suffix(f".tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, target=target, footprint=footprint)
        os.replace(temporary, sample.cache_path)
    return target, footprint


def filter_unsupervised_samples(dataset) -> list[str]:
    """Drop cache records with no room target before subset selection."""

    kept = []
    removed = []
    for sample in dataset.samples:
        target, footprint = load_or_rasterize(
            sample,
            image_size=dataset.image_size,
            taxonomy=dataset.taxonomy,
            write_cache=False,
        )
        key = f"{sample.relative_case}__{sample.floor_id}"
        if (target > 0).any() and (footprint > 0).any():
            kept.append(sample)
        else:
            removed.append(key)
    dataset.samples = kept
    return removed


def room_instance_targets(
    target: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    max_rooms: int,
) -> tuple:
    """Build padded instance targets in a data worker before H2D."""

    import torch

    from floorflow.heads.room_mask_decoder import room_mask_targets

    classes, masks, valid = room_mask_targets(
        torch.from_numpy(np.asarray(target).copy())[None],
        taxonomy,
        max_rooms=int(max_rooms),
    )
    item_valid = valid[0]
    return (
        classes[0],
        masks[0].to(torch.uint8),
        item_valid,
        item_valid.sum().to(torch.long),
    )


class CubiCasaFloorplanDataset:
    """PyTorch-compatible dataset without importing Torch at module import time."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        image_size: int = 256,
        cache_root: str | Path | None = None,
        augment: bool = False,
        limit: int | None = None,
        seed: int = 0,
        write_cache: bool = True,
        preload: bool = False,
        include_image: bool = False,
    ) -> None:
        self.cache_root = Path(cache_root) if cache_root else None
        taxonomy_path = self.cache_root / "taxonomy.json" if self.cache_root else None
        if taxonomy_path is not None and taxonomy_path.is_file():
            self.taxonomy = FloorplanTaxonomy.read(taxonomy_path)
        else:
            split_cases = read_split(Path(data_root) / f"{split}.txt")
            svg_paths = [Path(data_root) / case / "model.svg" for case in split_cases]
            self.taxonomy = build_taxonomy(scan_taxonomy_counts([path for path in svg_paths if path.is_file()]))
        self.samples = collect_samples(data_root, split, cache_root=cache_root, limit=limit)
        if not self.samples:
            raise FileNotFoundError(f"no model.svg files found for split={split!r} under {data_root}")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.write_cache = bool(write_cache)
        self.include_image = bool(include_image)
        self._targets: list[np.ndarray] | None = None
        self._footprints: list[np.ndarray] | None = None
        if preload:
            self.preload()

    def preload(self) -> None:
        if self._targets is not None and self._footprints is not None:
            return
        self._targets = []
        self._footprints = []
        for sample in self.samples:
            target, footprint = load_or_rasterize(
                sample,
                image_size=self.image_size,
                taxonomy=self.taxonomy,
                write_cache=self.write_cache,
            )
            self._targets.append(target)
            self._footprints.append(footprint)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = None
        if self.include_image:
            from floorflow.data.raster_control import load_cubicasa_raster

            if not sample.raster_path.is_file():
                raise FileNotFoundError(sample.raster_path)
            image = load_cubicasa_raster(
                sample.raster_path,
                image_size=self.image_size,
            )
        if self._targets is not None and self._footprints is not None:
            target = self._targets[index]
            footprint = self._footprints[index]
        else:
            target, footprint = load_or_rasterize(
                sample,
                image_size=self.image_size,
                taxonomy=self.taxonomy,
                write_cache=self.write_cache,
            )
        if self.augment:
            # Deterministic per item is not useful across epochs, so mix the global
            # random stream with the index. Worker processes inherit distinct seeds.
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
                if image is not None:
                    image = np.flip(image, axis=2)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
                if image is not None:
                    image = np.flip(image, axis=1)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
                if image is not None:
                    image = np.rot90(image, k=rotations, axes=(1, 2))
            if random.random() < 0.5:
                angle = random.uniform(-5.0, 5.0)
                scale = random.uniform(0.95, 1.05)
                radians = math.radians(angle)
                cosine, sine = math.cos(radians) / scale, math.sin(radians) / scale
                center = (self.image_size - 1) / 2.0
                tx = random.uniform(-4.0, 4.0)
                ty = random.uniform(-4.0, 4.0)
                coeffs = (
                    cosine,
                    sine,
                    center - cosine * center - sine * center + tx,
                    -sine,
                    cosine,
                    center + sine * center - cosine * center + ty,
                )
                target = np.asarray(
                    Image.fromarray(target).transform(
                        (self.image_size, self.image_size),
                        Image.Transform.AFFINE,
                        coeffs,
                        resample=Image.Resampling.NEAREST,
                        fillcolor=0,
                    )
                )
                footprint = np.asarray(
                    Image.fromarray(footprint * 255).transform(
                        (self.image_size, self.image_size),
                        Image.Transform.AFFINE,
                        coeffs,
                        resample=Image.Resampling.NEAREST,
                        fillcolor=0,
                    )
                )
                footprint = (footprint > 127).astype(np.uint8)
                if image is not None:
                    import cv2

                    matrix = np.asarray(
                        [
                            [cosine, sine, tx],
                            [-sine, cosine, ty],
                        ],
                        dtype=np.float32,
                    )
                    image = cv2.warpAffine(
                        np.moveaxis(image, 0, -1),
                        matrix,
                        (self.image_size, self.image_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    image = np.moveaxis(image, -1, 0)

        result = (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            int(sample.floor_id.rsplit("-", 1)[-1]),
            f"{sample.relative_case}__{sample.floor_id}",
        )
        if self.include_image:
            return (
                *result,
                torch.from_numpy(image.copy()).float(),
            )
        return result


class CubiCasaGeometryDataset:
    """Dataset that carries visible labels, room support, and occlusion masks."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        geometry_root: str | Path,
        *,
        augment: bool = False,
        limit: int | None = None,
    ) -> None:
        from floorflow.data.geometry_cache import geometry_cache_path, load_geometry_cache
        from floorflow.data.taxonomy import FloorplanTaxonomy

        self.split = split
        self.geometry_root = Path(geometry_root)
        self.taxonomy = FloorplanTaxonomy.read(self.geometry_root / "taxonomy.json")
        self.samples = collect_samples(
            data_root,
            split,
            limit=limit,
            floors=True,
        )
        self.paths = [
            geometry_cache_path(
                self.geometry_root,
                split,
                sample.relative_case,
                sample.floor_id,
            )
            for sample in self.samples
        ]
        self.augment = bool(augment)
        self.load_geometry_cache = load_geometry_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        payload = self.load_geometry_cache(self.paths[index])
        target = payload["target"]
        footprint = payload["footprint"]
        room_support = payload["room_support"]
        occlusion = payload["occlusion_mask"]
        if self.augment:
            if random.random() < 0.5:
                target, footprint, room_support, occlusion = (
                    np.flip(item, axis=1)
                    for item in (target, footprint, room_support, occlusion)
                )
            if random.random() < 0.5:
                target, footprint, room_support, occlusion = (
                    np.flip(item, axis=0)
                    for item in (target, footprint, room_support, occlusion)
                )
            rotations = random.randrange(4)
            if rotations:
                target, footprint, room_support, occlusion = (
                    np.rot90(item, k=rotations)
                    for item in (target, footprint, room_support, occlusion)
                )
        return (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            int(sample.floor_id.rsplit("-", 1)[-1]),
            f"{sample.relative_case}__{sample.floor_id}",
            torch.from_numpy(room_support.copy()).float(),
            torch.from_numpy(occlusion.copy()).float(),
        )


class CubiCasaRasterControlDataset(CubiCasaFloorplanDataset):
    """Return the native CubiCasa raster alongside semantic supervision."""

    def __init__(
        self,
        *args,
        image_mode: str = "native",
        include_room_targets: bool = False,
        room_target_limit: int = 24,
        **kwargs,
    ) -> None:
        if image_mode not in {"native", "line", "floor"}:
            raise ValueError("image_mode must be native, line, or floor")
        self.image_mode = image_mode
        self.include_room_targets = bool(include_room_targets)
        self.room_target_limit = int(room_target_limit)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        import torch

        from floorflow.data.raster_control import (
            crop_floor_masks,
            footprint_center,
            load_cubicasa_floor_raster,
            load_cubicasa_line_raster,
            load_cubicasa_raster,
        )
        from floorflow.data.svg import list_cubicasa_floors

        sample = self.samples[index]
        target, footprint = load_or_rasterize(
            sample,
            image_size=self.image_size,
            taxonomy=self.taxonomy,
            write_cache=self.write_cache,
        )
        if self.image_mode == "floor":
            center = footprint_center(footprint)
            target, footprint = crop_floor_masks(
                target,
                footprint,
                image_size=self.image_size,
            )
            floor_ids = list_cubicasa_floors(sample.svg_path)
            image = load_cubicasa_floor_raster(
                sample.raster_path,
                floor_index=floor_ids.index(str(sample.floor_id)),
                floor_count=len(floor_ids),
                target_center=center,
                image_size=self.image_size,
            )
        else:
            loader = (
                load_cubicasa_line_raster
                if self.image_mode == "line"
                else load_cubicasa_raster
            )
            image = loader(
                sample.raster_path,
                image_size=self.image_size,
            )
        if self.augment:
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
                image = np.flip(image, axis=2)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
                image = np.flip(image, axis=1)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
                image = np.rot90(image, k=rotations, axes=(1, 2))
        result = (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            torch.from_numpy(image.copy()).float(),
            int(sample.floor_id.rsplit("-", 1)[-1]),
            f"{sample.relative_case}__{sample.floor_id}",
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
