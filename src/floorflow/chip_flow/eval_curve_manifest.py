"""Frozen cross-seed validation curve manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.data_manifest import cache_fingerprint
from floorflow.chip_flow.train_evaluation import select_curve_indices
from floorflow.chip_flow.training_runtime import write_json_atomic


def build_eval_curve_manifest(
    store: PackedCaseStore,
    *,
    per_count: int,
    eval_seed: int,
    exclude_indices: set[int] | None = None,
) -> dict:
    indices = select_curve_indices(
        store,
        per_count=per_count,
        seed=eval_seed,
        exclude_indices=exclude_indices,
    )
    return {
        "version": "chip-eval-curve-manifest-v1",
        "per_count": int(per_count),
        "eval_seed": int(eval_seed),
        "excluded_indices": sorted(int(value) for value in (exclude_indices or ())),
        "indices": indices.tolist(),
        "block_counts": [
            int(store.arrays["n"][index]) for index in indices.tolist()
        ],
        "dev_fingerprint": cache_fingerprint(store.root),
    }


def load_eval_curve_manifest(
    path: str | Path,
    store: PackedCaseStore,
    *,
    verify_fingerprint: bool = True,
) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != "chip-eval-curve-manifest-v1":
        raise ValueError("unsupported evaluation curve manifest")
    indices = np.asarray(manifest.get("indices", []), dtype=np.int64)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(store):
        raise ValueError("evaluation curve indices are invalid")
    expected_counts = [
        int(store.arrays["n"][index]) for index in indices.tolist()
    ]
    if expected_counts != manifest.get("block_counts"):
        raise ValueError("evaluation curve block counts do not match the cache")
    if verify_fingerprint and cache_fingerprint(store.root) != manifest.get(
        "dev_fingerprint"
    ):
        raise ValueError("dev cache does not match evaluation curve manifest")
    return manifest


def write_eval_curve_manifest(path: str | Path, manifest: dict) -> None:
    write_json_atomic(path, manifest)


def resolve_eval_curve(
    store: PackedCaseStore,
    *,
    path: str | Path | None,
    per_count: int,
    eval_seed: int,
) -> tuple[np.ndarray, dict]:
    if path is not None:
        manifest = load_eval_curve_manifest(path, store)
    else:
        manifest = build_eval_curve_manifest(
            store,
            per_count=per_count,
            eval_seed=eval_seed,
        )
    return np.asarray(manifest["indices"], dtype=np.int64), manifest


__all__ = [
    "build_eval_curve_manifest",
    "load_eval_curve_manifest",
    "resolve_eval_curve",
    "write_eval_curve_manifest",
]
