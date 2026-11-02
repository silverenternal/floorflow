#!/usr/bin/env python3
"""Build the L0-L5 metric hierarchy and paper link-budget table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.metric_hierarchy import (
    build_metric_hierarchy,
    hierarchy_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--style-report", type=Path, required=True)
    parser.add_argument("--recursive-report", type=Path, required=True)
    parser.add_argument("--decision-audit", type=Path, required=True)
    parser.add_argument("--identifiability", type=Path, required=True)
    parser.add_argument("--diversity-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    report = build_metric_hierarchy(
        load(args.report),
        style_report=load(args.style_report),
        recursive_report=load(args.recursive_report),
        decision_audit=load(args.decision_audit),
        identifiability=load(args.identifiability),
        diversity_report=(
            load(args.diversity_report)
            if args.diversity_report is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        hierarchy_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["ranked_links"], indent=2))


if __name__ == "__main__":
    main()
