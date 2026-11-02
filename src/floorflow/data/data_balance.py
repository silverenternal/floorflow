"""Frequency-aware sampling for rare room and object labels."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_presence_weights(
    labels: list[np.ndarray],
    class_ids: tuple[int, ...],
    *,
    power: float = 0.5,
    maximum: float = 5.0,
) -> torch.Tensor:
    """Weight samples containing rare classes without duplicating pixels."""

    counts = Counter()
    for label_map in labels:
        present = set(int(value) for value in np.unique(label_map))
        counts.update(class_id for class_id in class_ids if class_id in present)
    frequencies = np.asarray([counts.get(class_id, 1) for class_id in class_ids], dtype=np.float64)
    inverse = (frequencies / max(1.0, frequencies.mean())).clip(min=1.0) ** (-float(power))
    weight_by_class = {
        class_id: min(float(maximum), float(weight))
        for class_id, weight in zip(class_ids, inverse)
    }
    sample_weights = []
    for label_map in labels:
        present = set(int(value) for value in np.unique(label_map))
        weight = max((weight_by_class[class_id] for class_id in class_ids if class_id in present), default=1.0)
        sample_weights.append(weight)
    return torch.as_tensor(sample_weights, dtype=torch.double)


def make_weighted_sampler(weights: torch.Tensor, num_samples: int | None = None) -> WeightedRandomSampler:
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples or len(weights),
        replacement=True,
    )


def read_dataset_labels(dataset) -> list[np.ndarray]:
    original = dataset.augment
    dataset.augment = False
    labels = []
    try:
        for index in range(len(dataset)):
            target = dataset[index][0]
            labels.append(target.numpy())
    finally:
        dataset.augment = original
    return labels


def cached_class_presence_weights(
    dataset,
    class_ids: tuple[int, ...],
    *,
    power: float = 0.5,
) -> torch.Tensor:
    """Cache rare-class sample weights inside the dataset cache root."""

    cache_root = getattr(dataset, "cache_root", None)
    if cache_root is None:
        return class_presence_weights(
            read_dataset_labels(dataset),
            class_ids,
            power=power,
        )
    digest = hashlib.sha256(
        (
            str(len(dataset))
            + ":"
            + str(float(power))
            + ":"
            + ",".join(str(value) for value in class_ids)
        ).encode("utf-8")
    ).hexdigest()[:16]
    path = Path(cache_root) / f"sample_weights_{digest}.npy"
    if path.is_file():
        values = np.load(path)
        if values.shape == (len(dataset),):
            return torch.from_numpy(values)
    values = class_presence_weights(
        read_dataset_labels(dataset),
        class_ids,
        power=power,
    ).numpy()
    temporary = path.with_suffix(f".tmp-{os.getpid()}.npy")
    np.save(temporary, values)
    os.replace(temporary, path)
    return torch.from_numpy(values)
