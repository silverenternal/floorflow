"""PyTorch dataset for plan-level skv4 JSONL records."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from floorflow.data.skv4 import rasterize_skv4_plan
from floorflow.data.taxonomy import FloorplanTaxonomy


class Skv4PlanDataset:
    def __init__(
        self,
        root: str | Path,
        split: str,
        taxonomy: FloorplanTaxonomy,
        *,
        image_size: int = 256,
        augment: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.taxonomy = taxonomy
        self.image_size = int(image_size)
        self.augment = bool(augment)
        path = self.root / f"{split}.jsonl"
        with path.open(encoding="utf-8") as source:
            self.plans = [json.loads(line) for line in source]
        if limit is not None:
            self.plans = self.plans[: int(limit)]
        if not self.plans:
            raise FileNotFoundError(f"no skv4 plans in {path}")

    def __len__(self) -> int:
        return len(self.plans)

    def __getitem__(self, index: int):
        import torch

        plan = self.plans[index]
        target, footprint = rasterize_skv4_plan(
            plan,
            self.taxonomy,
            self.image_size,
        )
        if self.augment:
            if random.random() < 0.5:
                target = np.flip(target, axis=1)
                footprint = np.flip(footprint, axis=1)
            if random.random() < 0.5:
                target = np.flip(target, axis=0)
                footprint = np.flip(footprint, axis=0)
            rotations = random.randrange(4)
            if rotations:
                target = np.rot90(target, k=rotations)
                footprint = np.rot90(footprint, k=rotations)
        return (
            torch.from_numpy(target.copy()).long(),
            torch.from_numpy(footprint.copy()).float(),
            0,
            str(plan["id"]),
        )


__all__ = ["Skv4PlanDataset"]
