#!/usr/bin/env python3
"""Validate FEVP v1.0 reports and optional case-row files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.fevp_protocol import (
    aggregate_case_rows,
    load_json,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--schema-root", type=Path, default=ROOT / "schema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema_paths = (
        args.schema_root / "fevp_report_v1.schema.json",
        args.schema_root / "fevp_case_row_v1.schema.json",
    )
    missing_schemas = [str(path) for path in schema_paths if not path.is_file()]
    if missing_schemas:
        raise FileNotFoundError(f"missing FEVP schemas: {missing_schemas}")
    for path in schema_paths:
        json.loads(path.read_text(encoding="utf-8"))
    report = load_json(args.report)
    errors = validate_report(
        report,
        require_complete=args.require_complete,
    )
    aggregate = None
    if args.cases is not None:
        payload = json.loads(args.cases.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload)
        if not isinstance(rows, list):
            errors.append("case file must be a list or contain a rows list")
        elif not errors:
            try:
                aggregate = aggregate_case_rows(rows)
            except ValueError as error:
                errors.append(str(error))
    output = {
        "status": "pass" if not errors else "fail",
        "report": str(args.report),
        "require_complete": args.require_complete,
        "errors": errors,
        "aggregate": aggregate,
    }
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
