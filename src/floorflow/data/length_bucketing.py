"""Deterministic length bucketing for variable-length token batches."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        bucket_multiplier: int = 8,
    ) -> None:
        if len(lengths) != len(set(range(len(lengths)))):
            raise ValueError("lengths must cover each dataset index")
        if batch_size <= 0 or bucket_multiplier <= 0:
            raise ValueError("batch_size and bucket_multiplier must be positive")
        self.lengths = tuple(int(value) for value in lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = self.batch_size * int(bucket_multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.lengths),
            generator=generator,
        ).tolist()
        indices.sort(key=lambda index: self.lengths[index])
        buckets = [
            indices[start : start + self.bucket_size]
            for start in range(0, len(indices), self.bucket_size)
        ]
        if self.shuffle:
            order = torch.randperm(
                len(buckets),
                generator=generator,
            ).tolist()
            buckets = [buckets[index] for index in order]
            for bucket in buckets:
                order = torch.randperm(
                    len(bucket),
                    generator=generator,
                ).tolist()
                bucket[:] = [bucket[index] for index in order]
        for bucket in buckets:
            for start in range(0, len(bucket), self.batch_size):
                yield bucket[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


__all__ = ["LengthBucketBatchSampler"]
