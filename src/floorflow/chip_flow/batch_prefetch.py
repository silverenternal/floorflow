"""Deterministic CPU materialization prefetch for chip training."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

import numpy as np
import torch

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.train_evaluation import (
    BalancedBatchSampler,
    augment_geometry,
)


def select_training_indices(
    store: PackedCaseStore,
    rng: np.random.Generator,
    *,
    batch_size: int,
    allowed_by_count: Mapping[int, np.ndarray] | None,
    sampler: BalancedBatchSampler | None,
) -> np.ndarray:
    if sampler is not None:
        _, indices = sampler.next_indices()
        return indices
    groups = allowed_by_count or store.by_block_count
    count = int(rng.choice(sorted(groups)))
    candidates = np.asarray(groups[count], dtype=np.int64)
    return rng.choice(
        candidates,
        size=batch_size,
        replace=len(candidates) < batch_size,
    )


class BufferedBatchStream:
    """Prefetch CPU gather while the current micro-batch is on the GPU."""

    def __init__(
        self,
        store: PackedCaseStore,
        rng: np.random.Generator,
        *,
        batch_size: int,
        device: torch.device,
        batches: int,
        geometry_augment: str = "none",
        geometry_augment_probability: float = 1.0,
        allowed_by_count: Mapping[int, np.ndarray] | None = None,
        sampler: BalancedBatchSampler | None = None,
        enabled: bool = True,
    ) -> None:
        if batches <= 0:
            raise ValueError("prefetch window must contain at least one batch")
        self.store = store
        self.rng = rng
        self.batch_size = int(batch_size)
        self.device = device
        self.batches = int(batches)
        self.geometry_augment = geometry_augment
        self.geometry_augment_probability = float(geometry_augment_probability)
        self.allowed_by_count = allowed_by_count
        self.sampler = sampler
        self.delivered = 0
        resident = getattr(store, "resident_device", None)
        self.direct_device = (
            resident is not None and torch.device(resident) == device
        )
        self.pending_indices: np.ndarray | None = None
        self.pending: Future | None = None
        self.executor = (
            ThreadPoolExecutor(max_workers=1)
            if enabled and not self.direct_device
            else None
        )

    def _submit(self) -> None:
        in_flight = 1 if self.pending_indices is not None else 0
        if self.delivered + in_flight >= self.batches:
            return
        self.pending_indices = select_training_indices(
            self.store,
            self.rng,
            batch_size=self.batch_size,
            allowed_by_count=self.allowed_by_count,
            sampler=self.sampler,
        )
        pin_memory = self.device.type == "cuda"
        batch_kwargs = {"device": "cpu"}
        if pin_memory:
            batch_kwargs["pin_memory"] = True
        self.pending = (
            self.executor.submit(
                self.store.batch,
                self.pending_indices,
                **batch_kwargs,
            )
            if self.executor is not None
            else None
        )

    def next(self) -> dict:
        if self.delivered >= self.batches:
            raise StopIteration
        if self.pending_indices is None:
            self._submit()
        if self.pending_indices is None:
            raise RuntimeError("batch prefetch has no pending indices")
        batch = (
            self.pending.result()
            if self.pending is not None
            else self.store.batch(
                self.pending_indices,
                device=self.device if self.direct_device else "cpu",
            )
        )
        indices = self.pending_indices
        self.pending = None
        self.pending_indices = None
        self.delivered += 1
        asynchronous = self.executor is not None and self.device.type == "cuda"
        batch = {
            key: (
                value
                if key == "n"
                else value.to(self.device, non_blocking=asynchronous)
            )
            for key, value in batch.items()
        }
        batch["_indices"] = indices
        batch = augment_geometry(
            batch,
            self.geometry_augment,
            self.rng,
            probability=self.geometry_augment_probability,
        )
        if self.delivered < self.batches:
            self._submit()
        return batch

    def __iter__(self):
        for _ in range(self.batches - self.delivered):
            yield self.next()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        self._submit()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


__all__ = ["BufferedBatchStream", "select_training_indices"]
