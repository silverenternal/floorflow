"""Train-split room-class adjacency statistics for graph semantic reasoning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_partition import build_room_partition_target


def accumulate_pair_counts(
    counts: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> np.ndarray:
    partition = build_room_partition_target(
        target.astype(np.uint8, copy=False),
        footprint.astype(np.uint8, copy=False),
        taxonomy,
    )
    for left, right in (
        (partition[:, :-1], partition[:, 1:]),
        (partition[:-1, :], partition[1:, :]),
    ):
        valid = (left >= 0) & (right >= 0) & (left != right)
        np.add.at(counts, (left[valid], right[valid]), 1.0)
        np.add.at(counts, (right[valid], left[valid]), 1.0)
    return counts


def normalize_pair_counts(
    counts: np.ndarray,
    smoothing: float = 0.1,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 0.0)
    prior = counts + float(smoothing)
    np.fill_diagonal(prior, 0.0)
    return prior / prior.sum(axis=1, keepdims=True).clip(min=1.0e-8)


def build_room_pair_prior(
    cache_root: str | Path,
    taxonomy: FloorplanTaxonomy,
    *,
    limit: int | None = None,
) -> dict:
    paths = sorted((Path(cache_root) / "train").rglob("*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    counts = np.zeros((len(taxonomy.room_ids), len(taxonomy.room_ids)))
    for path in paths:
        with np.load(path) as payload:
            accumulate_pair_counts(
                counts,
                payload["target"],
                payload["footprint"],
                taxonomy,
            )
    conditional = normalize_pair_counts(counts)
    digest = hashlib.sha256(
        counts.astype(np.float64).tobytes()
        + json.dumps(taxonomy.to_json(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "source_split": "train",
        "cases": len(paths),
        "room_ids": list(taxonomy.room_ids),
        "room_names": [
            taxonomy.class_names[class_id]
            for class_id in taxonomy.room_ids
        ],
        "counts": counts.tolist(),
        "conditional": conditional.tolist(),
        "fingerprint": digest,
        "best_of_n_used": False,
    }


def write_room_pair_prior(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "accumulate_pair_counts",
    "build_room_pair_prior",
    "normalize_pair_counts",
    "write_room_pair_prior",
]
