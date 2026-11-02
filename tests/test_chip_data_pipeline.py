from __future__ import annotations

import numpy as np
import torch

from floorflow.chip_flow.batch_prefetch import BufferedBatchStream
from floorflow.chip_flow.complexity_sampler import (
    ComplexityBatchSampler,
    ProgressiveComplexityBatchSampler,
    complexity_strata,
)
from floorflow.chip_flow.proposals import (
    TokenOTCache,
    token_ot_reorder,
    token_ot_reorder_batch,
)


class _Store:
    def __init__(self) -> None:
        self.arrays = {
            "n": np.asarray([21, 21, 35, 35, 61, 61, 81, 81]),
            "n_b2b": np.asarray([4, 12, 10, 40, 20, 80, 30, 120]),
            "n_pins": np.asarray([10, 30, 20, 60, 30, 90, 50, 150]),
        }
        self.by_block_count = {
            count: np.flatnonzero(self.arrays["n"] == count)
            for count in sorted(set(self.arrays["n"].tolist()))
        }

    def __len__(self) -> int:
        return len(self.arrays["n"])

    def indices(self, block_count: int | None = None) -> np.ndarray:
        return (
            np.arange(len(self))
            if block_count is None
            else self.by_block_count[int(block_count)]
        )

    def batch(self, indices: np.ndarray, *, device="cpu") -> dict:
        count = int(self.arrays["n"][indices[0]])
        return {
            "n": count,
            "areas": torch.ones(len(indices), count),
            "marker": torch.as_tensor(np.asarray(indices), dtype=torch.float32),
        }


def test_buffered_prefetch_matches_synchronous_order() -> None:
    store = _Store()
    allowed = store.by_block_count
    kwargs = {
        "batch_size": 2,
        "device": torch.device("cpu"),
        "batches": 3,
        "allowed_by_count": allowed,
    }
    with BufferedBatchStream(
        store, np.random.default_rng(7), enabled=True, **kwargs
    ) as stream:
        prefetched = list(stream)
    with BufferedBatchStream(
        store, np.random.default_rng(7), enabled=False, **kwargs
    ) as stream:
        direct = list(stream)
    for left, right in zip(prefetched, direct, strict=True):
        assert np.array_equal(left["_indices"], right["_indices"])
        torch.testing.assert_close(left["marker"], right["marker"])


def test_complexity_sampler_uses_multiple_strata() -> None:
    store = _Store()
    assert len(complexity_strata(store, store.indices())) > 1
    sampler = ComplexityBatchSampler(store, batch_size=2, seed=11)
    batches = [sampler.next_indices()[1] for _ in range(6)]
    assert len({tuple(sorted(values.tolist())) for values in batches}) > 1


def test_progressive_complexity_sampler_expands_block_bands() -> None:
    store = _Store()
    sampler = ProgressiveComplexityBatchSampler(
        store,
        batch_size=2,
        seed=13,
        total_steps=100,
        final_fraction=0.5,
    )
    sampler.set_step(0)
    early = np.concatenate([sampler.next_indices()[1] for _ in range(4)])
    assert np.all(store.arrays["n"][early] <= 40)
    sampler.set_step(100)
    late = np.concatenate([sampler.next_indices()[1] for _ in range(20)])
    assert np.any(store.arrays["n"][late] >= 61)


def test_batched_ot_reorder_matches_reference() -> None:
    source, target = torch.randn(3, 6, 3), torch.randn(3, 6, 3)
    expected = torch.stack(
        [token_ot_reorder(source[row], target[row]) for row in range(3)]
    )
    torch.testing.assert_close(
        token_ot_reorder_batch(source, target, workers=2),
        expected,
    )


def test_token_ot_cache_reuses_identical_batch_rows() -> None:
    source, target = torch.randn(2, 5, 3), torch.randn(2, 5, 3)
    cache = TokenOTCache(max_entries=4)
    first = token_ot_reorder_batch(
        source,
        target,
        case_indices=np.asarray([3, 8]),
        cache=cache,
    )
    second = token_ot_reorder_batch(
        source,
        target,
        case_indices=np.asarray([3, 8]),
        cache=cache,
    )
    torch.testing.assert_close(second, first)
    assert cache.hits == 2
    assert cache.misses == 2
