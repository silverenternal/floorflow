#!/usr/bin/env python3
"""Analyze multiscale opening losses by detector failure category."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.opening_failure_strata import (
    analyze_opening_failure_strata,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/opening_multiscale_failure_strata_val491_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
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
    report = analyze_opening_failure_strata(
        _load(args.baseline),
        _load(args.candidate),
        _load(args.oracle),
        iterations=args.bootstrap,
    )
    report["sources"] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "oracle": str(args.oracle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "overall": report["overall"],
            "by_diagnosis": report["by_diagnosis"],
            "by_target_count": report["by_target_count"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
