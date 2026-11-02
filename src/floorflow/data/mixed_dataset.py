"""Controlled real/synthetic sampling without altering either dataset."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler


@dataclass(frozen=True)
class MixtureSpec:
    real_count: int
    synthetic_count: int
    real_fraction: float
    samples_per_epoch: int


@dataclass(frozen=True)
class MixtureCurriculum:
    """Start on real data, then ramp synthetic exposure to a cap."""

    real_only_epochs: int = 1
    ramp_epochs: int = 2
    final_real_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.real_only_epochs < 0 or self.ramp_epochs < 0:
            raise ValueError("curriculum epochs must be non-negative")
        if not 0.0 <= float(self.final_real_fraction) <= 1.0:
            raise ValueError("final_real_fraction must be in [0, 1]")

    def real_fraction(self, epoch: int) -> float:
        epoch = max(0, int(epoch))
        if epoch < self.real_only_epochs:
            return 1.0
        if self.ramp_epochs == 0:
            return float(self.final_real_fraction)
        progress = min(
            1.0,
            (epoch - self.real_only_epochs + 1) / float(self.ramp_epochs),
        )
        return 1.0 + progress * (float(self.final_real_fraction) - 1.0)


def make_real_synthetic_mixture(
    real_dataset,
    synthetic_dataset,
    *,
    real_fraction: float = 1.0,
    samples_per_epoch: int | None = None,
    seed: int = 17,
) -> tuple[ConcatDataset, WeightedRandomSampler, MixtureSpec]:
    real_count = len(real_dataset)
    synthetic_count = len(synthetic_dataset)
    if real_count + synthetic_count == 0:
        raise ValueError("mixture requires at least one sample")
    fraction = min(1.0, max(0.0, float(real_fraction)))
    total = int(samples_per_epoch or real_count + synthetic_count)
    real_weight = fraction / max(1, real_count)
    synthetic_weight = (1.0 - fraction) / max(1, synthetic_count)
    weights = [
        real_weight if index < real_count else synthetic_weight
        for index in range(real_count + synthetic_count)
    ]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=max(1, total),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    spec = MixtureSpec(
        real_count=real_count,
        synthetic_count=synthetic_count,
        real_fraction=fraction,
        samples_per_epoch=max(1, total),
    )
    return (
        ConcatDataset([real_dataset, synthetic_dataset]),
        sampler,
        spec,
    )
