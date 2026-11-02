"""Bind typed-channel validation rows to atom proxy metric contracts."""

from __future__ import annotations


def _oriented(value: float, direction: str, transform: str) -> float:
    result = abs(value) if transform == "absolute" else value
    return -result if direction == "minimize" else result


def _contrast_sensitivity(rows: list[dict], pair) -> list[dict]:
    by_checkpoint: dict[str, list[dict]] = {}
    for row in rows:
        by_checkpoint.setdefault(row["checkpoint"], []).append(row)
    if len(by_checkpoint) < 2:
        return []
    baseline_name = min(by_checkpoint)
    baseline = by_checkpoint[baseline_name]
    baseline_proxy = sum(row["proxy"] for row in baseline) / len(baseline)
    baseline_canonical = sum(row["canonical"] for row in baseline) / len(baseline)
    baseline_proxy_oriented = _oriented(
        baseline_proxy,
        pair.proxy_direction,
        pair.proxy_transform,
    )
    baseline_canonical_oriented = _oriented(
        baseline_canonical,
        pair.canonical_direction,
        pair.canonical_transform,
    )
    for checkpoint in sorted(by_checkpoint)[1:]:
        values = by_checkpoint[checkpoint]
        proxy = sum(row["proxy"] for row in values) / len(values)
        canonical = sum(row["canonical"] for row in values) / len(values)
        if (
            _oriented(proxy, pair.proxy_direction, pair.proxy_transform)
            < baseline_proxy_oriented
            and _oriented(
                canonical,
                pair.canonical_direction,
                pair.canonical_transform,
            )
            < baseline_canonical_oriented
        ):
            return [
                {
                    "case_id": "cohort_mean",
                    "attack": f"checkpoint_contrast:{checkpoint}",
                    "proxy_baseline": baseline_proxy,
                    "proxy_attack": proxy,
                    "canonical_baseline": baseline_canonical,
                    "canonical_attack": canonical,
                }
            ]
    return []


def build_measurements(rows: list[dict], pairs) -> list[dict]:
    """Bind available per-case rows to configured atom metric pairs."""

    measurements = []
    for pair in pairs:
        selected = []
        for row in rows:
            proxy = row.get("proxy_metrics", {})
            canonical = row.get("canonical_metrics", {})
            if (
                pair.proxy_metric not in proxy
                or pair.canonical_metric not in canonical
            ):
                continue
            selected.append(
                {
                    "seed": row["seed"],
                    "checkpoint": row["checkpoint"],
                    "case_id": row["case_id"],
                    "proxy": float(proxy[pair.proxy_metric]),
                    "canonical": float(canonical[pair.canonical_metric]),
                }
            )
        if not selected:
            continue
        checkpoints = sorted({row["checkpoint"] for row in selected})
        measurements.append(
            {
                "id": pair.id,
                "atom_id": pair.atom_id,
                "proxy_metric": pair.proxy_metric,
                "canonical_metric": pair.canonical_metric,
                "proxy_direction": pair.proxy_direction,
                "canonical_direction": pair.canonical_direction,
                "proxy_transform": pair.proxy_transform,
                "canonical_transform": pair.canonical_transform,
                "proxy_tolerance": pair.proxy_tolerance,
                "canonical_tolerance": pair.canonical_tolerance,
                "baseline_checkpoint": checkpoints[0],
                "target_used_at_inference": False,
                "rows": selected,
                "sensitivity": _contrast_sensitivity(selected, pair),
            }
        )
    return measurements


__all__ = ["build_measurements"]
