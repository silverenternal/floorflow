#!/usr/bin/env python3
"""Audit raw-to-legalized candidate rank preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.legalization_survival import (
    DEFAULT_VARIANTS,
    legalization_survival,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_legalization_survival_v1.json"),
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    rows = []
    for path in args.report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report_rows = payload.get("rows")
        if not isinstance(report_rows, list) or not report_rows:
            raise ValueError(f"report has no rows: {path}")
        reports.append(str(path))
        for row in report_rows:
            rows.append({"index": int(row["index"]), **row})
    result = legalization_survival(
        rows,
        variants=DEFAULT_VARIANTS,
        tolerance=args.tolerance,
    )
    result["reports"] = reports
    result["contract"] = (
        "Lower-is-better pairwise ordering on refined versus legalized "
        "candidate views; no official target is recomputed."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
