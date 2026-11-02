"""Deterministic stratified selection for legalizer-response sidecars."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from floorflow.chip_flow.complexity_sampler import complexity_strata


def select_response_cases(
    store,
    indices: Iterable[int],
    *,
    max_cases: int,
    seed: int,
) -> list[int]:
    """Select roughly equal cases from each complexity stratum."""

    rows = np.asarray(sorted({int(index) for index in indices}), dtype=np.int64)
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    if max_cases > len(rows):
        raise ValueError("max_cases exceeds the available case count")
    strata = complexity_strata(store, rows)
    rng = np.random.default_rng(seed)
    shuffled = {
        key: values[rng.permutation(len(values))]
        for key, values in sorted(strata.items())
    }
    base, remainder = divmod(max_cases, len(shuffled))
    selected: list[int] = []
    overflow = 0
    for position, values in enumerate(shuffled.values()):
        quota = base + int(position < remainder)
        take = min(quota, len(values))
        selected.extend(int(value) for value in values[:take])
        overflow += quota - take
    if overflow:
        used = set(selected)
        for values in shuffled.values():
            for value in values:
                index = int(value)
                if index not in used:
                    selected.append(index)
                    used.add(index)
                    overflow -= 1
                    if overflow == 0:
                        break
            if overflow == 0:
                break
    return sorted(selected)


__all__ = ["select_response_cases"]
