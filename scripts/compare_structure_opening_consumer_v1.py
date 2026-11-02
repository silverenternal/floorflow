#!/usr/bin/env python3
"""Compare identical structure plans with and without a learned opening layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.opening_consumer_comparison import (
    compare_opening_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/opening_interval_consumer_val491_paired_v1.json",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_used_at_inference") is not False:
        raise ValueError(f"target-free contract missing in {path}")
    if payload.get("test_opened") is not False:
        raise ValueError(f"test was opened in {path}")
    return payload


def main() -> None:
    args = parse_args()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    comparison = compare_opening_reports(
        baseline,
        candidate,
        iterations=args.iterations,
        seed=args.seed,
    )
    report = {
        "schema_version": "opening-interval-consumer-comparison-v1",
        "protocol": "predicted_control",
        "target_used_at_inference": False,
        "test_opened": False,
        "cases": comparison["cases"],
        "baseline": {
            "path": str(args.baseline),
            "config": baseline.get("config"),
            "summary": baseline["summary"]["blend"],
        },
        "candidate": {
            "path": str(args.candidate),
            "config": candidate.get("config"),
            "summary": candidate["summary"]["blend"],
        },
        **comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
