"""Deterministic nearest-contour memory for room-partition priors."""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from floorflow.representation.core import contour_descriptor
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.data.taxonomy import FloorplanTaxonomy


def _partition_cache_worker(
    index_path: tuple[int, str],
    taxonomy: FloorplanTaxonomy,
) -> tuple[int, np.ndarray]:
    index, path = index_path
    with np.load(path) as payload:
        target = payload["target"].astype(np.uint8, copy=False)
        footprint = payload["footprint"].astype(np.uint8, copy=False)
    partition = build_room_partition_target(target, footprint, taxonomy)
    return index, partition.astype(np.int16, copy=False)


class ContourPartitionMemory:
    """Retrieve one training partition by contour descriptor similarity."""

    def __init__(
        self,
        paths: list[Path],
        top_k: int = 1,
        metric: str = "cosine",
    ) -> None:
        if not paths:
            raise ValueError("contour memory requires at least one training sample")
        self.paths = [Path(path) for path in paths]
        self.top_k = max(1, int(top_k))
        self.metric = str(metric)
        descriptors = []
        masks = []
        for path in self.paths:
            with np.load(path) as payload:
                descriptors.append(contour_descriptor(payload["footprint"]))
                masks.append(payload["footprint"].astype(np.float32))
        self.descriptors = np.asarray(descriptors, dtype=np.float64)
        self.descriptors /= np.linalg.norm(
            self.descriptors,
            axis=1,
            keepdims=True,
        ).clip(min=1.0e-8)
        mask_tensor = torch.from_numpy(np.stack(masks)).float()[:, None]
        self.masks = torch.nn.functional.adaptive_avg_pool2d(
            mask_tensor,
            (32, 32),
        ).flatten(1).numpy()
        digest = hashlib.sha256()
        for path in self.paths:
            digest.update(str(path).encode("utf-8"))
            digest.update(b"\0")
        digest.update(self.descriptors.astype(np.float32).tobytes())
        digest.update(self.masks.astype(np.float32).tobytes())
        self.fingerprint = digest.hexdigest()
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._partition_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    @classmethod
    def from_cache(
        cls,
        cache_root: str | Path,
        top_k: int = 1,
        metric: str = "cosine",
    ) -> "ContourPartitionMemory":
        paths = sorted((Path(cache_root) / "train").rglob("*.npz"))
        return cls(paths, top_k=top_k, metric=metric)

    def retrieve_top_indices(
        self,
        footprints: np.ndarray | torch.Tensor,
        *,
        top_k: int | None = None,
        exclude_indices: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        values = footprints.detach().cpu().numpy() if torch.is_tensor(footprints) else np.asarray(footprints)
        if values.ndim == 2:
            values = values[None]
        if values.size and float(values.max()) > 1.0:
            values = values.astype(np.float32) / 255.0
        else:
            values = values.astype(np.float32)
        if self.metric in {"mask_iou", "hybrid"}:
            masks = torch.from_numpy(values.astype(np.float32))[:, None]
            queries = torch.nn.functional.adaptive_avg_pool2d(
                masks,
                (32, 32),
            ).flatten(1).numpy()
            bank = self.masks
            intersection = queries @ bank.T
            denominator = (
                queries.sum(axis=1, keepdims=True)
                + bank.sum(axis=1)[None, :]
                - intersection
            )
            mask_scores = intersection / np.clip(denominator, 1.0e-6, None)
        if self.metric in {"cosine", "hybrid"}:
            descriptors = np.stack([contour_descriptor(item) for item in values]).astype(np.float64)
            descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True).clip(min=1.0e-8)
            cosine_scores = descriptors @ self.descriptors.T
        if self.metric == "mask_iou":
            scores = mask_scores
        elif self.metric == "cosine":
            scores = cosine_scores
        else:
            def normalize_rows(matrix: np.ndarray) -> np.ndarray:
                minimum = matrix.min(axis=1, keepdims=True)
                maximum = matrix.max(axis=1, keepdims=True)
                return (matrix - minimum) / np.clip(maximum - minimum, 1.0e-8, None)

            scores = 0.5 * (normalize_rows(mask_scores) + normalize_rows(cosine_scores))
        if exclude_indices is not None:
            excluded = (
                exclude_indices.detach().cpu().numpy()
                if torch.is_tensor(exclude_indices)
                else np.asarray(exclude_indices)
            )
            scores[np.arange(scores.shape[0]), excluded.astype(np.int64)] = -np.inf
        count = min(int(top_k or self.top_k), scores.shape[1])
        indices = np.argpartition(-scores, count - 1, axis=1)[:, :count]
        ordering = np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1)
        return np.take_along_axis(indices, ordering, axis=1)

    def retrieve_indices(self, footprints: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.retrieve_top_indices(footprints, top_k=1)[:, 0]

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        index = int(index)
        if index not in self._cache:
            with np.load(self.paths[index]) as payload:
                self._cache[index] = (
                    payload["target"].astype(np.uint8, copy=True),
                    payload["footprint"].astype(np.uint8, copy=True),
                )
        return self._cache[index]

    def _load_partition(
        self,
        index: int,
        taxonomy: FloorplanTaxonomy,
    ) -> np.ndarray:
        index = int(index)
        if index not in self._partition_cache:
            target, footprint = self._load(index)
            partition = build_room_partition_target(target, footprint, taxonomy)
            self._partition_cache[index] = partition.astype(np.int16, copy=False)
        return self._partition_cache[index]

    def precompute_partitions(
        self,
        taxonomy: FloorplanTaxonomy,
        *,
        workers: int = 8,
    ) -> int:
        """Build all dense partition priors once before training batches."""

        missing = [
            (index, str(path))
            for index, path in enumerate(self.paths)
            if index not in self._partition_cache
        ]
        if not missing:
            return 0
        if workers <= 1:
            for index, path in missing:
                with np.load(path) as payload:
                    target = payload["target"].astype(np.uint8, copy=False)
                    footprint = payload["footprint"].astype(np.uint8, copy=False)
                self._partition_cache[index] = build_room_partition_target(
                    target,
                    footprint,
                    taxonomy,
                ).astype(np.int16, copy=False)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for index, partition in executor.map(
                    _partition_cache_worker,
                    missing,
                    [taxonomy] * len(missing),
                ):
                    self._partition_cache[index] = partition
        return len(missing)

    def retrieve_labels(
        self,
        footprints: np.ndarray | torch.Tensor,
        *,
        top_k: int | None = None,
        exclude_indices: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        indices = self.retrieve_top_indices(
            footprints,
            top_k=top_k,
            exclude_indices=exclude_indices,
        )
        labels = [self._load(int(index))[0] for index in indices[:, 0]]
        return torch.from_numpy(np.stack(labels)).long()

    def retrieve_partitions(
        self,
        footprints: np.ndarray | torch.Tensor,
        taxonomy: FloorplanTaxonomy,
    ) -> torch.Tensor:
        indices = self.retrieve_indices(footprints)
        parts = []
        for index in indices:
            target, footprint = self._load(int(index))
            parts.append(build_room_partition_target(target, footprint, taxonomy))
        return torch.from_numpy(np.stack(parts))

    def retrieve_logits(
        self,
        footprints: np.ndarray | torch.Tensor,
        taxonomy: FloorplanTaxonomy,
        device: torch.device | str,
        *,
        top_k: int | None = None,
        exclude_indices: np.ndarray | torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        top_indices = self.retrieve_top_indices(
            footprints,
            top_k=top_k,
            exclude_indices=exclude_indices,
        )
        spatial_shape = self._load(0)[1].shape
        votes = torch.zeros(
            (top_indices.shape[0], len(taxonomy.room_ids), *spatial_shape),
            dtype=torch.float32,
            device=device,
        )
        for rank in range(top_indices.shape[1]):
            parts = []
            for index in top_indices[:, rank]:
                parts.append(self._load_partition(int(index), taxonomy))
            partition = torch.from_numpy(np.stack(parts)).to(
                device=device,
                dtype=torch.long,
            )
            valid = partition >= 0
            one_hot = torch.nn.functional.one_hot(
                partition.clamp_min(0),
                len(taxonomy.room_ids),
            ).permute(0, 3, 1, 2).float()
            votes += one_hot * valid[:, None].float()
        votes /= float(top_indices.shape[1])
        logits = 2.0 * votes - 1.0
        if class_weights is not None:
            weights = class_weights.to(device=device, dtype=logits.dtype)
            if weights.numel() != logits.shape[1]:
                raise ValueError("retrieval class weights must match room-class count")
            logits = logits * weights[None, :, None, None]
        return logits
