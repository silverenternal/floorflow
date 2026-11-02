"""Label-distribution diagnostics for real and synthetic floorplan caches."""

from __future__ import annotations

import numpy as np


def normalized_pixel_distribution(
    counts: np.ndarray,
    class_ids: list[int],
) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)[class_ids]
    total = values.sum()
    return values / total if total > 0.0 else np.zeros_like(values)


def distribution_summary(real: np.ndarray, synthetic: np.ndarray) -> dict:
    real = np.asarray(real, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    if real.shape != synthetic.shape:
        raise ValueError("real and synthetic distributions must have equal shape")
    total_variation = 0.5 * np.abs(real - synthetic).sum()
    jensen_shannon = 0.0
    for value in (real, synthetic):
        mask = value > 0.0
        if mask.any():
            midpoint = 0.5 * (real + synthetic)
            jensen_shannon += 0.5 * np.sum(
                value[mask] * np.log(value[mask] / midpoint[mask])
            )
    return {
        "total_variation": float(total_variation),
        "jensen_shannon": float(max(0.0, jensen_shannon)),
        "cosine_similarity": float(
            np.dot(real, synthetic)
            / max(1.0e-12, np.linalg.norm(real) * np.linalg.norm(synthetic))
        ),
    }


def class_alignment_rows(
    class_names: tuple[str, ...],
    class_ids: list[int],
    real_counts: np.ndarray,
    synthetic_counts: np.ndarray,
) -> list[dict]:
    real = normalized_pixel_distribution(real_counts, class_ids)
    synthetic = normalized_pixel_distribution(synthetic_counts, class_ids)
    rows = []
    for index, class_id in enumerate(class_ids):
        rows.append(
            {
                "class": class_names[class_id],
                "real_fraction": float(real[index]),
                "synthetic_fraction": float(synthetic[index]),
                "fraction_ratio": (
                    float(synthetic[index] / real[index])
                    if real[index] > 0.0
                    else None
                ),
                "missing_in_synthetic": bool(
                    real[index] > 0.0 and synthetic[index] == 0.0
                ),
                "missing_in_real": bool(
                    synthetic[index] > 0.0 and real[index] == 0.0
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(
            np.log(max(1.0e-8, row["synthetic_fraction"]))
            - np.log(max(1.0e-8, row["real_fraction"]))
        ),
        reverse=True,
    )
