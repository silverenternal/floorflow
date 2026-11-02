#!/usr/bin/env python3
"""Gate the style-controlled generation claim on the frozen candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style-report",
        type=Path,
        default=ROOT
        / "outputs/reports/final_retrieval_free_style_val8_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/style_claim_gate_v1.json",
    )
    parser.add_argument("--separation-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = read(args.style_report)
    separation = float(report["style_separation"])
    report_out = {
        "status": "pass" if separation >= args.separation_threshold else "fail",
        "claim": "style-controlled generation",
        "metric": "style_separation",
        "value": separation,
        "threshold": args.separation_threshold,
        "decision": (
            "keep"
            if separation >= args.separation_threshold
            else "drop-from-headline-and-report-as-limitation"
        ),
        "best_of_n_used": report.get("best_of_n_used", False),
        "target_used_for_generation": report.get(
            "target_used_for_generation", False
        ),
        "source": str(args.style_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_out, indent=2))


if __name__ == "__main__":
    main()
