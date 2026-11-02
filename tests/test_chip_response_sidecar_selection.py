from __future__ import annotations

import numpy as np

from floorflow.chip_flow.response_sidecar_selection import select_response_cases


class _Store:
    def __init__(self) -> None:
        count = 80
        self.arrays = {
            "n": np.arange(count) % 80 + 21,
            "n_b2b": np.arange(count) % 12,
            "n_pins": np.arange(count) % 17,
        }


def test_stratified_response_selection_is_unique_and_deterministic() -> None:
    store = _Store()
    indices = list(range(80))
    first = select_response_cases(
        store,
        indices,
        max_cases=24,
        seed=19,
    )
    second = select_response_cases(
        store,
        indices,
        max_cases=24,
        seed=19,
    )
    assert first == second
    assert len(first) == 24
    assert len(set(first)) == 24
    assert min(first) >= 0 and max(first) < 80
