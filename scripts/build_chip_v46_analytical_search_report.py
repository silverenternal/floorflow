#!/usr/bin/env python3
"""Combine matched-flow analytical-search arms into one decision report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.analytical_feedback import summarize_matched_flow_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-n1",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_candidate_n1_positioned_all.json",
    )
    parser.add_argument(
        "--candidate-n2",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_candidate_n2_positioned_all.json",
    )
    parser.add_argument(
        "--control-n1",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_control_n1_positioned_all.json",
    )
    parser.add_argument(
        "--control-n2",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_control_n2_positioned_all.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_report.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    inputs = {
        "candidate_n1": args.candidate_n1,
        "candidate_n2": args.candidate_n2,
        "control_n1": args.control_n1,
        "control_n2": args.control_n2,
    }
    reports = {name: load(path) for name, path in inputs.items()}
    output = summarize_matched_flow_search(reports)
    output["inputs"] = {
        name: str(path.relative_to(ROOT)) for name, path in inputs.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output)}, indent=2))
    if output["status"] == "reject_contract":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
