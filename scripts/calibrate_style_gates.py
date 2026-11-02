#!/usr/bin/env python3
"""Calibrate deployment style gates from validation-only reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.style_gate_calibration import select_style_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--no-style-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    no_style = json.loads(args.no_style_report.read_text(encoding="utf-8"))
    minimum_coverage = float(no_style["distribution"]["coverage"])
    report_paths = sorted(Path().glob(args.reports))
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in report_paths
    ]
    selected = select_style_gates(
        reports,
        minimum_coverage=minimum_coverage,
    )
    expected = {str(index) for index in range(int(bank["num_styles"]))}
    if set(selected) != expected:
        raise SystemExit(
            f"style gate calibration missing styles: {sorted(expected - set(selected))}"
        )
    output = {
        **bank,
        "style_gates": [
            selected[str(index)]["gate"]
            for index in range(int(bank["num_styles"]))
        ],
        "gate_calibration": {
            "split": "val",
            "minimum_coverage": minimum_coverage,
            "no_style_report": str(args.no_style_report),
            "reports": [str(path) for path in report_paths],
            "selected": selected,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
