#!/usr/bin/env python3
"""Build the fail-closed FP-H multi-seed hard-guarantee report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.multiseed_validation import build_multiseed_report


def parse_run(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        raise argparse.ArgumentTypeError("run must use SEED=METRICS_PATH")
    return int(seed), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_h_multiseed_test503_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_multiseed_report(
        args.run,
        bootstrap_iterations=args.bootstrap,
        source_paths=args.source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
