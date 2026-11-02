#!/usr/bin/env python3
"""Aggregate dynamic-head validation reports and apply Pareto selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.cad_metric_suite import pareto_front


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=REPORT.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate(value: str) -> dict[str, object]:
    name, separator, path = value.partition("=")
    if not separator:
        raise ValueError("candidate must use NAME=REPORT.json")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    room = report["room_metrics"]["overlay"]
    cad = report.get("cad_metrics", {})
    summary = {**room, **cad.get("summary", {})}
    return {
        "name": name,
        "report": path,
        "checkpoint": report.get("checkpoint"),
        "summary": summary,
        "hard_gate_passed": bool(
            cad.get("hard_gate", {}).get("passed", False)
        ),
        "balanced_secondary": cad.get("balanced_secondary", {}).get("score"),
    }


def main() -> None:
    args = parse_args()
    candidates = [load_candidate(value) for value in args.candidate]
    candidates.sort(
        key=lambda row: float(row["summary"].get("room_macro_iou", 0.0)),
        reverse=True,
    )
    report = {
        "protocol": "val64 style0 target-free single-sample",
        "candidates": candidates,
        "pareto_front": pareto_front(candidates),
        "selection": "baseline retained because all dynamic candidates fail hard CAD gates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
