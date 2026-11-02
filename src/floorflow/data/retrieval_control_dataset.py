"""Leave-one-out train retrieval controls for residual layout learning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from floorflow.data.control_cache import resize_control
from floorflow.heads.room_program import room_presence_target
from floorflow.representation.contour_memory import ContourPartitionMemory


class RetrievalControlDataset(Dataset):
    """Attach train-only nearest-contour layouts to each floorplan sample."""

    def __init__(
        self,
        base: Dataset,
        cache_root: str | Path,
        split: str,
        *,
        top_k: int = 1,
        metric: str = "cosine",
        control_size: int = 128,
    ) -> None:
        self.base = base
        self.split = str(split)
        self.taxonomy = base.taxonomy
        self.control_size = max(2, int(control_size))
        self.memory = ContourPartitionMemory.from_cache(
            cache_root,
            top_k=top_k,
            metric=metric,
        )
        sample_paths = [
            Path(sample.cache_path).resolve()
            for sample in base.samples
        ]
        memory_paths = [
            Path(path).resolve()
            for path in self.memory.paths
        ]
        self._memory_index = {
            path: index for index, path in enumerate(memory_paths)
        }
        self._exclude_indices = None
        if self.split == "train":
            missing = [
                path for path in sample_paths
                if path not in self._memory_index
            ]
            if missing:
                raise ValueError(
                    f"{len(missing)} training samples are absent from "
                    "the contour memory"
                )
            self._exclude_indices = [
                self._memory_index[path] for path in sample_paths
            ]

    def __len__(self) -> int:
        return len(self.base)

    def _retrieve_control(
        self,
        footprint: torch.Tensor,
        index: int,
    ) -> dict[str, torch.Tensor]:
        exclusion = None
        if self._exclude_indices is not None:
            exclusion = torch.tensor(
                [self._exclude_indices[index]],
                dtype=torch.long,
            )
        raw_prior = self.memory.retrieve_logits(
            footprint[None].float(),
            self.taxonomy,
            "cpu",
            exclude_indices=exclusion,
        )
        coarse_logits = resize_control(raw_prior, self.control_size)
        partition = raw_prior.argmax(dim=1)
        room_ids = torch.tensor(self.taxonomy.room_ids)
        rooms = room_ids[partition]
        occupied = raw_prior.amax(dim=1) > -0.999
        rooms = torch.where(
            occupied & footprint[None].bool(),
            rooms,
            torch.zeros_like(rooms),
        )
        return {
            "room_logits": coarse_logits[0],
            "program": room_presence_target(rooms, self.taxonomy)[0],
        }

    def __getitem__(self, index: int):
        batch = self.base[index]
        target, footprint, floor_id, name = batch[:4]
        del target, floor_id, name
        return (*batch, self._retrieve_control(footprint, index))


__all__ = ["RetrievalControlDataset"]
