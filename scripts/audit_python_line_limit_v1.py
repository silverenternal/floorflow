#!/usr/bin/env python3
"""Audit the repository engineering limit of 600 lines per Python file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_line_limit_v17.json",
    )
    parser.add_argument(
        "--exclude-token",
        action="append",
        default=[],
        help="Exclude paths containing this case-insensitive token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(
        path
        for directory in ("scripts", "src", "tests")
        for path in (ROOT / directory).rglob("*.py")
        if ".venv" not in path.parts
        and not any(
            token.lower() in str(path.relative_to(ROOT)).lower()
            for token in args.exclude_token
        )
    )
    rows = {
        str(path.relative_to(ROOT)): len(
            path.read_text(encoding="utf-8").splitlines()
        )
        for path in files
    }
    violations = {
        path: lines for path, lines in rows.items() if lines > LIMIT
    }
    report = {
        "status": "complete" if not violations else "failed",
        "limit": LIMIT,
        "files": len(rows),
        "exclude_tokens": args.exclude_token,
        "max_lines": max(rows.values()),
        "max_file": max(rows, key=rows.get),
        "violations": violations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
