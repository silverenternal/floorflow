"""Dataset-level class frequencies for stable long-tail supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def count_cached_classes(cache_root: str | Path, num_classes: int) -> np.ndarray:
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for path in sorted(Path(cache_root).rglob("*.npz")):
        with np.load(path) as payload:
            counts += np.bincount(payload["target"].ravel(), minlength=num_classes)[:num_classes]
    return counts


def positive_weights(
    class_counts: np.ndarray,
    *,
    max_weight: float = 20.0,
) -> np.ndarray:
    """Return per-foreground-channel BCE positive weights."""

    counts = np.asarray(class_counts, dtype=np.float64)[1:]
    foreground_total = max(1.0, float(np.asarray(class_counts, dtype=np.float64).sum()))
    ratio = (foreground_total - counts).clip(min=1.0) / counts.clip(min=1.0)
    return np.sqrt(ratio).clip(1.0, float(max_weight)).astype(np.float32)


def write_class_frequency_report(
    cache_root: str | Path,
    *,
    class_names: tuple[str, ...] | list[str],
    max_weight: float = 20.0,
) -> Path:
    cache_root = Path(cache_root)
    counts = count_cached_classes(cache_root, len(class_names))
    weights = positive_weights(counts, max_weight=max_weight)
    report = {
        "class_names": list(class_names),
        "pixel_counts": counts.tolist(),
        "pixel_fractions": (counts / max(1, counts.sum())).tolist(),
        "positive_weights": weights.tolist(),
        "max_positive_weight": float(max_weight),
    }
    path = cache_root / "class_frequency.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(cache_root / "positive_weights.npy", weights)
    return path


def load_positive_weights(cache_root: str | Path, num_target_channels: int, max_weight: float) -> np.ndarray:
    path = Path(cache_root) / "positive_weights.npy"
    if path.is_file():
        weights = np.load(path).astype(np.float32)
        if weights.shape == (int(num_target_channels),):
            return weights
    counts_path = Path(cache_root) / "class_frequency.json"
    if counts_path.is_file():
        payload = json.loads(counts_path.read_text(encoding="utf-8"))
        return positive_weights(np.asarray(payload["pixel_counts"]), max_weight=max_weight)
    raise FileNotFoundError(f"class frequency report missing under {cache_root}")
