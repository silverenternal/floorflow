"""Train-only contour retrieval used as a coarse room-layout prior."""

from __future__ import annotations

import numpy as np
import torch

from floorflow.representation.core import contour_descriptor
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.data.taxonomy import FloorplanTaxonomy


class ContourRoomRetriever:
    def __init__(
        self,
        targets: list[np.ndarray],
        footprints: list[np.ndarray],
        taxonomy: FloorplanTaxonomy,
        top_k: int = 1,
    ) -> None:
        if len(targets) != len(footprints):
            raise ValueError("targets and footprints must align")
        self.taxonomy = taxonomy
        self.top_k = max(1, int(top_k))
        self.targets = targets
        self.partitions = np.stack(
            [build_room_partition_target(target, footprint, taxonomy) for target, footprint in zip(targets, footprints)]
        ).astype(np.int8)
        descriptors = np.stack([contour_descriptor(footprint) for footprint in footprints]).astype(np.float64)
        self.descriptor_dim = descriptors.shape[-1]
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True).clip(min=1.0e-8)
        self.descriptors = descriptors

    def _nearest(
        self,
        footprints: np.ndarray,
        exclude_indices: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        queries = np.asarray(footprints, dtype=np.float64)
        if queries.ndim == 2:
            queries = queries[None]
        if queries.ndim != 3:
            raise ValueError("footprints must have shape [H,W] or [B,H,W]")
        if queries.shape[-1] != self.descriptor_dim:
            queries = np.stack([contour_descriptor(item) for item in queries]).astype(np.float64)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1.0e-8)
        scores = queries @ self.descriptors.T
        if exclude_indices is not None:
            rows = np.arange(scores.shape[0])
            scores[rows, np.asarray(exclude_indices, dtype=np.int64)] = -np.inf
        count = min(int(top_k or self.top_k), scores.shape[1])
        indices = np.argpartition(-scores, count - 1, axis=1)[:, :count]
        selected_scores = np.take_along_axis(scores, indices, axis=1)
        weights = np.exp(selected_scores - selected_scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True).clip(min=1.0e-12)
        return indices, weights, selected_scores

    def retrieve(
        self,
        footprints: np.ndarray,
        exclude_indices: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Return local room indices [B,H,W] and never consult test targets."""

        indices, weights, _scores = self._nearest(footprints, exclude_indices, top_k)
        selected = torch.from_numpy(self.partitions[indices].astype(np.int64))
        votes = torch.zeros(
            (indices.shape[0], len(self.taxonomy.room_ids), *selected.shape[-2:]),
            dtype=torch.float32,
        )
        weight_tensor = torch.from_numpy(weights.astype(np.float32))
        for rank in range(selected.shape[1]):
            partition = selected[:, rank].unsqueeze(1)
            valid = partition >= 0
            votes.scatter_add_(
                1,
                partition.clamp_min(0),
                valid.float() * weight_tensor[:, rank, None, None, None],
            )
        return votes.argmax(dim=1).numpy()

    def retrieve_labels(
        self,
        footprints: np.ndarray,
        exclude_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one fixed nearest train label map per query."""

        indices, _weights, _scores = self._nearest(footprints, exclude_indices)
        return np.stack([self.targets[int(index[0])] for index in indices])

    def retrieve_partitions(
        self,
        footprints: np.ndarray,
        exclude_indices: np.ndarray | None = None,
        top_k: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, weights, _scores = self._nearest(footprints, exclude_indices, top_k)
        return self.partitions[indices].astype(np.int64), weights.astype(np.float32)

    def nearest_similarity(
        self,
        footprints: np.ndarray,
        exclude_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        _indices, _weights, scores = self._nearest(footprints, exclude_indices)
        return scores[:, 0]


def prior_one_hot(
    prior: np.ndarray | torch.Tensor,
    num_classes: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    tensor = prior if isinstance(prior, torch.Tensor) else torch.from_numpy(prior)
    tensor = tensor.to(device=device, dtype=torch.long)
    return torch.nn.functional.one_hot(tensor, num_classes).permute(0, 3, 1, 2).float()
