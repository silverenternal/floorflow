from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floorflow.chip_flow.data_manifest import build_data_manifest, verify_data_manifest
from floorflow.chip_flow.eval_curve_manifest import (
    build_eval_curve_manifest,
    load_eval_curve_manifest,
    write_eval_curve_manifest,
)


class _Store:
    def __init__(self, root: Path, counts: list[int]) -> None:
        self.root = root
        self.arrays = {"n": np.asarray(counts)}
        self.by_block_count = {
            count: np.flatnonzero(self.arrays["n"] == count)
            for count in sorted(set(counts))
        }

    def __len__(self) -> int:
        return len(self.arrays["n"])

    def indices(self, block_count: int | None = None) -> np.ndarray:
        return (
            np.arange(len(self))
            if block_count is None
            else self.by_block_count[block_count]
        )


def _cache(root: Path, seed: int) -> None:
    root.mkdir()
    (root / "metadata.json").write_text(f'{{"seed": {seed}}}\n')
    np.save(root / "n.npy", np.asarray([21, 22, 21], dtype=np.int16))


def test_data_manifest_detects_cache_changes(tmp_path) -> None:
    train, dev = tmp_path / "train", tmp_path / "dev"
    _cache(train, 1)
    _cache(dev, 2)
    manifest = build_data_manifest(train, dev)
    verify_data_manifest(manifest, train, dev)
    (dev / "metadata.json").write_text('{"seed": 3}\n')
    with pytest.raises(ValueError, match="does not match"):
        verify_data_manifest(manifest, train, dev)


def test_eval_curve_is_frozen_by_eval_seed(tmp_path) -> None:
    root = tmp_path / "dev"
    _cache(root, 6)
    store = _Store(root, [21, 21, 21, 22, 22, 22])
    manifest = build_eval_curve_manifest(store, per_count=1, eval_seed=0)
    path = tmp_path / "curve.json"
    write_eval_curve_manifest(path, manifest)
    assert load_eval_curve_manifest(path, store)["indices"] == manifest["indices"]


def test_eval_curve_can_exclude_prior_cases(tmp_path) -> None:
    root = tmp_path / "dev"
    _cache(root, 6)
    store = _Store(root, [21, 21, 21, 22, 22, 22])
    first = build_eval_curve_manifest(store, per_count=1, eval_seed=0)
    second = build_eval_curve_manifest(
        store,
        per_count=1,
        eval_seed=0,
        exclude_indices=set(first["indices"]),
    )
    assert set(first["indices"]).isdisjoint(second["indices"])
