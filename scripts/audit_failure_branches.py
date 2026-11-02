#!/usr/bin/env python3
"""Summarize failure and near-success patterns across training JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("outputs/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=2)
    return parser.parse_args()


def load_history(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_key(records: list[dict]) -> str | None:
    for key in ("room_macro_iou", "presence_f1"):
        if any(finite(record.get(key)) for record in records):
            return key
    return None


def loss_key(records: list[dict]) -> str | None:
    for key in ("train_loss", "loss", "train_total", "train_total_loss"):
        if any(finite(record.get(key)) for record in records):
            return key
    return None


def summarize_branch(name: str, records: list[dict]) -> dict:
    metric = metric_key(records)
    loss = loss_key(records)
    series = [
        (index, float(record[metric]))
        for index, record in enumerate(records)
        if metric and finite(record.get(metric))
    ]
    losses = [
        float(record[loss])
        for record in records
        if loss and finite(record.get(loss))
    ]
    best_index, best_value = max(series, key=lambda item: item[1])
    last_value = series[-1][1]
    first_value = series[0][1]
    anomalies = []
    if best_index == len(records) - 1 and len(records) >= 3:
        anomalies.append("best_at_last_epoch")
    if best_value - last_value > 0.005:
        anomalies.append("post_best_degradation")
    if losses and len(losses) == len(series):
        loss_change = (losses[-1] - losses[0]) / max(abs(losses[0]), 1.0e-9)
        metric_change = last_value - first_value
        if loss_change < -0.2 and metric_change < 0:
            anomalies.append("loss_improves_metric_regresses")
    if len(series) >= 3:
        rounded = [round(value, 6) for _, value in series]
        repeated = Counter(rounded).most_common(1)[0][1]
        if repeated >= 3 and len(set(losses)) > 1:
            anomalies.append("validation_metric_stalled")
    if best_value < 0.02:
        anomalies.append("near_zero_metric")
    invalid = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, float) and not math.isfinite(value)
        }
    )
    if invalid:
        anomalies.append("nonfinite_metric")
    return {
        "branch": name,
        "epochs": len(records),
        "metric": metric,
        "first": first_value,
        "best": best_value,
        "best_epoch": best_index,
        "last": last_value,
        "post_best_drop": best_value - last_value,
        "loss": loss,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "anomalies": anomalies,
        "nonfinite_keys": invalid,
    }


def main() -> None:
    args = parse_args()
    branches = []
    for path in sorted(args.train_root.glob("*/train.jsonl")):
        records = load_history(path)
        if len(records) >= args.min_history and metric_key(records):
            branches.append(summarize_branch(path.parent.name, records))
    branches.sort(key=lambda row: row["best"], reverse=True)
    report = {
        "branch_count": len(branches),
        "top_near_success": branches[:20],
        "undertrained": [
            row for row in branches if "best_at_last_epoch" in row["anomalies"]
        ],
        "loss_metric_divergence": [
            row
            for row in branches
            if "loss_improves_metric_regresses" in row["anomalies"]
        ],
        "post_best_degradation": [
            row
            for row in branches
            if "post_best_degradation" in row["anomalies"]
        ],
        "stalled_validation": [
            row
            for row in branches
            if "validation_metric_stalled" in row["anomalies"]
        ],
        "near_zero": [
            row for row in branches if "near_zero_metric" in row["anomalies"]
        ],
        "nonfinite": [
            row
            for row in branches
            if "nonfinite_metric" in row["anomalies"]
        ],
        "caveats": [
            "High target-derived-control branches are upper bounds, not contour-only results.",
            "probe train IoU is diagnostic only; validation IoU is the acceptance metric.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "branch_count": report["branch_count"],
                "top": branches[:10],
                "counts": {
                    key: len(report[key])
                    for key in (
                        "undertrained",
                        "loss_metric_divergence",
                        "post_best_degradation",
                        "stalled_validation",
                        "near_zero",
                        "nonfinite",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
