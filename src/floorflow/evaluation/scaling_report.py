"""Data-budget scaling curves and the train/val sweet-spot verdict."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _mean_metric(row: Mapping[str, object], names: Sequence[str]) -> float:
    values = [
        float(row[name])
        for name in names
        if isinstance(row.get(name), (int, float))
    ]
    return sum(values) / len(values) if values else 0.0


def _knee(
    points: Sequence[tuple[int, float]],
    *,
    gain_threshold: float,
) -> int | None:
    """Return the first size whose next doubling gains less than threshold."""

    ordered = sorted(points)
    for index in range(len(ordered) - 1):
        size, value = ordered[index]
        _next_size, next_value = ordered[index + 1]
        if next_value - value < float(gain_threshold):
            return int(size)
    return int(ordered[-1][0]) if ordered else None


def build_scaling_report(
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]] | None = None,
    *,
    metrics: Sequence[str] = (
        "room_macro_iou",
        "fpeval_wall_f1",
        "fpeval_opening_f1",
    ),
    gain_threshold: float = 0.01,
    max_composite_bias: float = 0.01,
) -> dict[str, object]:
    """Aggregate train-size and val-size sweeps into a sweet-spot verdict."""

    train_curve = []
    for row in sorted(train_rows, key=lambda item: int(item["train_size"])):
        train_curve.append(
            {
                "train_size": int(row["train_size"]),
                "val_size": int(row.get("val_size", 0)),
                "composite": (
                    float(row["composite"])
                    if "composite" in row
                    else _mean_metric(row, metrics)
                ),
                **{
                    name: row.get(name)
                    for name in metrics
                    if name in row
                },
            }
        )
    train_knee = _knee(
        [(row["train_size"], row["composite"]) for row in train_curve],
        gain_threshold=gain_threshold,
    )
    last_gain = (
        train_curve[-1]["composite"] - train_curve[-2]["composite"]
        if len(train_curve) >= 2
        else 0.0
    )
    val_curve = []
    for row in sorted(val_rows or [], key=lambda item: int(item["val_size"])):
        val_curve.append(
            {
                "val_size": int(row["val_size"]),
                "composite": (
                    float(row["composite"])
                    if "composite" in row
                    else _mean_metric(row, metrics)
                ),
                "metric_std": float(row.get("metric_std", 0.0)),
                "rank_stability": float(row.get("rank_stability", 0.0)),
            }
        )
    full_composite = val_curve[-1]["composite"] if val_curve else 0.0
    stable = [
        row
        for row in val_curve
        if row["rank_stability"] >= 0.98
        and abs(row["composite"] - full_composite) <= float(max_composite_bias)
    ]
    val_sweet = (
        min(stable, key=lambda row: row["val_size"])["val_size"]
        if stable
        else (val_curve[0]["val_size"] if val_curve else None)
    )
    return {
        "status": "complete" if train_curve else "incomplete",
        "metrics": list(metrics),
        "gain_threshold": float(gain_threshold),
        "max_composite_bias": float(max_composite_bias),
        "train_curve": train_curve,
        "val_curve": val_curve,
        "train_knee": train_knee,
        "last_doubling_gain": float(last_gain),
        "still_data_limited": bool(last_gain >= float(gain_threshold)),
        "sweet_spot": {
            "train_size": train_knee,
            "val_size": val_sweet,
        },
    }


__all__ = ["build_scaling_report"]
