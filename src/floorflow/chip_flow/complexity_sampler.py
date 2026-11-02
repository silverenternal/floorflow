"""Multi-dimensional complexity strata sampler for chip training."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.train_evaluation import BalancedBatchSampler


def complexity_strata(
    store: PackedCaseStore,
    indices: np.ndarray,
) -> dict[int, np.ndarray]:
    rows = np.asarray(indices, dtype=np.int64)
    counts = store.arrays["n"][rows].astype(np.float64)
    b2b_density = store.arrays["n_b2b"][rows] / np.maximum(counts, 1.0)
    pin_density = store.arrays["n_pins"][rows] / np.maximum(counts, 1.0)
    b2b_high = b2b_density >= np.median(b2b_density)
    pin_high = pin_density >= np.median(pin_density)
    band = np.where(
        counts <= 40,
        0,
        np.where(counts <= 60, 1, np.where(counts <= 80, 2, 3)),
    )
    stratum = band * 4 + b2b_high.astype(np.int64) * 2 + pin_high.astype(np.int64)
    return {
        int(value): rows[stratum == value]
        for value in np.unique(stratum)
    }


class ComplexityBatchSampler(BalancedBatchSampler):
    """Cycle multi-dimensional complexity strata without replacement."""

    def __init__(
        self,
        store: PackedCaseStore,
        *,
        indices: np.ndarray | None = None,
        allowed_by_count: Mapping[int, np.ndarray] | None = None,
        batch_size: int,
        seed: int,
    ) -> None:
        if indices is None:
            indices = (
                store.indices()
                if allowed_by_count is None
                else np.concatenate(list(allowed_by_count.values()))
            )
        groups = complexity_strata(store, indices)
        super().__init__(groups, batch_size=batch_size, seed=seed)
        self.indices = np.asarray(indices, dtype=np.int64)


class ProgressiveComplexityBatchSampler(ComplexityBatchSampler):
    """Expand block-count bands gradually while balancing graph density."""

    def __init__(
        self,
        store: PackedCaseStore,
        *,
        indices: np.ndarray | None = None,
        allowed_by_count: Mapping[int, np.ndarray] | None = None,
        batch_size: int,
        seed: int,
        total_steps: int,
        final_fraction: float = 0.5,
    ) -> None:
        super().__init__(
            store,
            indices=indices,
            allowed_by_count=allowed_by_count,
            batch_size=batch_size,
            seed=seed,
        )
        if total_steps <= 0 or not 0.0 < final_fraction <= 1.0:
            raise ValueError("invalid progressive complexity schedule")
        self.total_steps = int(total_steps)
        self.final_fraction = float(final_fraction)
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = max(0, int(step))

    def _active_counts(self) -> tuple[int, ...]:
        progress = min(
            1.0,
            self.current_step / max(1, round(self.total_steps * self.final_fraction)),
        )
        maximum_band = int(progress * 3.0)
        return tuple(
            key for key in self.counts if key // 4 <= maximum_band
        )

    def next_indices(self) -> tuple[int, np.ndarray]:
        active = self._active_counts()
        if not active:
            raise RuntimeError("progressive sampler has no active strata")
        group = active[self.cursor % len(active)]
        self.cursor += 1
        queue = self.queues[group]
        while len(queue) < self.batch_size:
            queue.extend(
                int(value) for value in self.rng.permutation(self.groups[group])
            )
        indices = np.asarray(queue[: self.batch_size], dtype=np.int64)
        del queue[: self.batch_size]
        return group, indices


def make_chip_batch_sampler(
    mode: str,
    store: PackedCaseStore,
    allowed_by_count: Mapping[int, np.ndarray] | None,
    *,
    batch_size: int,
    seed: int,
    total_steps: int = 1,
) -> BalancedBatchSampler | None:
    if mode == "random":
        return None
    if mode == "balanced_stream":
        groups = allowed_by_count or {
            int(count): np.asarray(indices, dtype=np.int64)
            for count, indices in store.by_block_count.items()
        }
        return BalancedBatchSampler(groups, batch_size=batch_size, seed=seed)
    if mode == "complexity_tail_stream":
        return ComplexityBatchSampler(
            store,
            allowed_by_count=allowed_by_count,
            batch_size=batch_size,
            seed=seed,
        )
    if mode == "progressive_complexity":
        return ProgressiveComplexityBatchSampler(
            store,
            allowed_by_count=allowed_by_count,
            batch_size=batch_size,
            seed=seed,
            total_steps=total_steps,
        )
    raise ValueError(f"unknown sampling mode: {mode}")


__all__ = [
    "ComplexityBatchSampler",
    "ProgressiveComplexityBatchSampler",
    "complexity_strata",
    "make_chip_batch_sampler",
]
