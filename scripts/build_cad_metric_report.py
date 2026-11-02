#!/usr/bin/env python3
"""Build the unified CAD metric report from existing per-case rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.cad_metric_suite import build_cad_metric_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-key", default="cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.per_case.read_text(encoding="utf-8"))
    rows = payload[args.cases_key] if args.cases_key else payload
    if not isinstance(rows, list):
        raise TypeError("per-case source must contain a case list")
    report = build_cad_metric_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": report["protocol"]["cases"],
                "hard_gate_passed": report["hard_gate"]["passed"],
                "balanced_secondary": report["balanced_secondary"]["score"],
                "missing_metrics": report["missing_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
