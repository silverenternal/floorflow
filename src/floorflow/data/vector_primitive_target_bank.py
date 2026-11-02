"""Canonical vector targets indexed by predicted-control case names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from floorflow.data.vector_geometry_dataset import VectorGeometryDataset

TARGET_KEYS = (
    "wall_endpoints",
    "wall_exists",
    "wall_thickness",
    "wall_exterior",
    "opening_centers",
    "opening_exists",
    "opening_widths",
    "opening_classes",
)


def case_name_from_geometry(relative: Path) -> str:
    """Convert ``category/case/Floor_N.geometry.npz`` to a case id."""

    category = relative.parts[-3]
    case_id = relative.parts[-2]
    floor_id = relative.parts[-1].removesuffix(".geometry.npz")
    return f"{category}__{case_id}__{floor_id.replace('_', '-')}"


def opening_host_targets(
    wall_endpoints: torch.Tensor,
    wall_exists: torch.Tensor,
    opening_centers: torch.Tensor,
    opening_exists: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest host-wall indices and normalized interval centers."""

    hosts = torch.zeros(
        opening_exists.shape,
        dtype=torch.long,
    )
    intervals = torch.zeros_like(opening_centers[..., 0])
    walls = wall_endpoints.detach().cpu().numpy()
    wall_mask = wall_exists.detach().cpu().numpy() > 0.5
    centers = opening_centers.detach().cpu().numpy()
    opening_mask = opening_exists.detach().cpu().numpy() > 0.5
    for batch in range(walls.shape[0]):
        wall_ids = np.flatnonzero(wall_mask[batch])
        if wall_ids.size == 0:
            continue
        starts = walls[batch, wall_ids, 0]
        ends = walls[batch, wall_ids, 1]
        deltas = ends - starts
        lengths_sq = np.maximum((deltas**2).sum(axis=-1), 1.0e-9)
        for opening in np.flatnonzero(opening_mask[batch]):
            point = centers[batch, opening]
            fraction = np.clip(
                ((point - starts) * deltas).sum(axis=-1) / lengths_sq,
                0.0,
                1.0,
            )
            projected = starts + fraction[:, None] * deltas
            distance = np.linalg.norm(projected - point[None], axis=-1)
            local = int(distance.argmin())
            hosts[batch, opening] = int(wall_ids[local])
            intervals[batch, opening] = float(fraction[local])
    return hosts, intervals


class VectorPrimitiveTargetBank:
    """Load canonical vector targets without re-vectorizing raster labels."""

    def __init__(
        self,
        geometry_root: str | Path,
        split: str,
        *,
        image_size: int = 256,
        max_walls: int = 64,
        max_openings: int = 48,
    ) -> None:
        self.root = Path(geometry_root)
        self.split = split
        self.dataset = VectorGeometryDataset(
            self.root,
            split,
            image_size=image_size,
            max_walls=max_walls,
            max_openings=max_openings,
        )
        self.index = {
            case_name_from_geometry(
                path.relative_to(self.root / self.split)
            ): item
            for item, path in enumerate(self.dataset.paths)
        }

    def target(
        self,
        name: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if name not in self.index:
            raise KeyError(f"missing canonical vector target for {name}")
        item = self.dataset[self.index[name]]
        targets = {
            key: item[key].to(device)
            for key in TARGET_KEYS
        }
        hosts, intervals = opening_host_targets(
            targets["wall_endpoints"][None],
            targets["wall_exists"][None],
            targets["opening_centers"][None],
            targets["opening_exists"][None],
        )
        targets["opening_hosts"] = hosts[0].to(device)
        targets["opening_intervals"] = intervals[0].to(device)
        return targets

    def batch_targets(
        self,
        names: tuple[str, ...] | list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        rows = [self.target(name, device) for name in names]
        return {
            key: torch.stack([row[key] for row in rows])
            for key in rows[0]
        }


__all__ = [
    "TARGET_KEYS",
    "VectorPrimitiveTargetBank",
    "case_name_from_geometry",
    "opening_host_targets",
]
