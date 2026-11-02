#!/usr/bin/env python3
"""Select the model-first sprint winner from val-only evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--minimum-relative-gain", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    baseline_iou = float(baseline["aggregate"]["quality_mean"])
    rows = []
    for path in args.reports:
        report = load(path)
        quality = float(report["aggregate"]["quality_mean"])
        relative_gain = (quality - baseline_iou) / max(baseline_iou, 1.0e-8)
        rows.append(
            {
                "path": str(path),
                "checkpoint": report.get("checkpoint"),
                "quality_mean": quality,
                "coverage_worst": float(
                    report["aggregate"]["coverage_worst"]
                ),
                "relative_gain": relative_gain,
                "eligible": relative_gain >= args.minimum_relative_gain,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible or rows
    winner = max(pool, key=lambda row: row["quality_mean"])
    result = {
        "baseline": str(args.baseline),
        "baseline_quality_mean": baseline_iou,
        "minimum_relative_gain": args.minimum_relative_gain,
        "candidates": rows,
        "winner": winner,
        "status": "eligible_winner" if eligible else "no_candidate_above_gate",
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
