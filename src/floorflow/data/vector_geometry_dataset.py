"""Padded vector-primitive targets from the CubiCasa geometry cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floorflow.data.geometry_cache import (
    geometry_from_payload,
    load_geometry_cache,
    unpack_polygons,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


def _resample_polygon(points: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        return np.zeros((count, 2), dtype=np.float32)
    closed = np.concatenate([values, values[:1]], axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 1.0e-6:
        return np.zeros((count, 2), dtype=np.float32)
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    result[:, 0] = np.interp(samples, cumulative, closed[:, 0])
    result[:, 1] = np.interp(samples, cumulative, closed[:, 1])
    return result


class VectorGeometryDataset:
    """Load geometry cache files as padded room/wall/opening targets."""

    def __init__(
        self,
        geometry_root: str | Path,
        split: str,
        *,
        image_size: int = 256,
        max_rooms: int = 32,
        max_walls: int = 64,
        max_openings: int = 48,
        polygon_points: int = 16,
        limit: int | None = None,
    ) -> None:
        self.geometry_root = Path(geometry_root)
        self.taxonomy = FloorplanTaxonomy.read(self.geometry_root / "taxonomy.json")
        self.paths = sorted(
            (self.geometry_root / split).rglob("*.geometry.npz")
        )
        if limit is not None:
            self.paths = self.paths[: int(limit)]
        self.image_size = int(image_size)
        self.max_rooms = int(max_rooms)
        self.max_walls = int(max_walls)
        self.max_openings = int(max_openings)
        self.polygon_points = int(polygon_points)
        opening_lookup = {
            int(class_id): index
            for index, class_id in enumerate(self.taxonomy.opening_ids)
        }
        self.opening_lookup = opening_lookup

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        import torch

        path = self.paths[index]
        payload = load_geometry_cache(path)
        geometry = geometry_from_payload(payload)
        scale = float(self.image_size)
        room_polygons = np.zeros(
            (self.max_rooms, self.polygon_points, 2),
            dtype=np.float32,
        )
        room_exists = np.zeros(self.max_rooms, dtype=np.float32)
        room_classes = np.zeros(self.max_rooms, dtype=np.int64)
        polygons = unpack_polygons(geometry.rooms)
        room_lookup = {
            int(class_id): local
            for local, class_id in enumerate(self.taxonomy.room_ids)
        }
        room_count = min(len(polygons), len(geometry.room_class_ids), self.max_rooms)
        for item in range(room_count):
            room_polygons[item] = _resample_polygon(
                polygons[item] / scale,
                self.polygon_points,
            )
            room_exists[item] = 1.0
            room_classes[item] = room_lookup.get(
                int(geometry.room_class_ids[item]),
                0,
            )
        wall_endpoints = np.zeros(
            (self.max_walls, 2, 2),
            dtype=np.float32,
        )
        wall_exists = np.zeros(self.max_walls, dtype=np.float32)
        wall_thickness = np.zeros(self.max_walls, dtype=np.float32)
        wall_exterior = np.zeros(self.max_walls, dtype=np.float32)
        wall_count = min(len(geometry.wall_centerlines), self.max_walls)
        wall_endpoints[:wall_count] = geometry.wall_centerlines[:wall_count] / scale
        wall_exists[:wall_count] = 1.0
        wall_thickness[:wall_count] = geometry.wall_thickness[:wall_count] / scale
        wall_exterior[:wall_count] = np.asarray(
            [str(value) == "exterior" for value in geometry.wall_types[:wall_count]],
            dtype=np.float32,
        )
        opening_centers = np.zeros(
            (self.max_openings, 2),
            dtype=np.float32,
        )
        opening_exists = np.zeros(self.max_openings, dtype=np.float32)
        opening_widths = np.zeros(self.max_openings, dtype=np.float32)
        opening_classes = np.zeros(self.max_openings, dtype=np.int64)
        opening_count = min(len(geometry.opening_centers), self.max_openings)
        opening_centers[:opening_count] = (
            geometry.opening_centers[:opening_count] / scale
        )
        opening_exists[:opening_count] = 1.0
        opening_widths[:opening_count] = (
            geometry.opening_widths[:opening_count] / scale
        )
        opening_classes[:opening_count] = np.asarray(
            [
                self.opening_lookup.get(int(class_id), 0)
                for class_id in geometry.opening_class_ids[:opening_count]
            ],
            dtype=np.int64,
        )
        floor_id = path.stem.removesuffix(".geometry")
        floor_number = int(floor_id.rsplit("_", 1)[-1])
        return {
            "target": torch.from_numpy(geometry.target.astype(np.int64)),
            "footprint": torch.from_numpy(geometry.footprint.astype(np.float32)),
            "floor_id": floor_number,
            "room_polygons": torch.from_numpy(room_polygons),
            "room_exists": torch.from_numpy(room_exists),
            "room_classes": torch.from_numpy(room_classes),
            "wall_endpoints": torch.from_numpy(wall_endpoints),
            "wall_exists": torch.from_numpy(wall_exists),
            "wall_thickness": torch.from_numpy(wall_thickness),
            "wall_exterior": torch.from_numpy(wall_exterior),
            "opening_centers": torch.from_numpy(opening_centers),
            "opening_exists": torch.from_numpy(opening_exists),
            "opening_widths": torch.from_numpy(opening_widths),
            "opening_classes": torch.from_numpy(opening_classes),
            "name": str(path.relative_to(self.geometry_root)),
        }


__all__ = ["VectorGeometryDataset"]
