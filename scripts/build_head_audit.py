#!/usr/bin/env python3
"""Summarize per-head training losses from a JSONL training log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.train_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("empty training log")
    components: dict[str, list[float]] = {}
    for record in records:
        train = record.get("train", record)
        for name, value in train.items():
            if name.endswith("_loss") and isinstance(value, (int, float)):
                components.setdefault(name, []).append(float(value))
    rows = []
    for name, values in sorted(components.items()):
        values_array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "head": name,
                "epochs": len(values),
                "first": float(values_array[0]),
                "last": float(values_array[-1]),
                "delta": float(values_array[-1] - values_array[0]),
                "nonzero_fraction": float(np.mean(np.abs(values_array) > 0.0)),
                "finite": bool(np.isfinite(values_array).all()),
            }
        )
    report = {
        "train_log": str(args.train_log),
        "records": len(records),
        "heads": rows,
        "zero_or_invalid": [
            row["head"]
            for row in rows
            if not row["finite"] or row["nonzero_fraction"] < 0.5
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
