#!/usr/bin/env python3
"""Replay legalizer policies on frozen refined chip layouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.legalizer_counterfactual import (
    compare_legalizer_policies,
    policies_from_config,
)
from scripts.evaluate_chip_official import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_legalizer_ablation_v1.json",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    case_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = case_payload.get("cases") if isinstance(case_payload, dict) else case_payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases file requires a non-empty cases list")
    policies = policies_from_config(config)
    rows = [
        compare_legalizer_policies(case, policies, score_case)
        for case in cases
    ]
    report = {
        "status": "complete",
        "config": str(args.config),
        "cases": len(rows),
        "variants": list(policies),
        "rows": rows,
        "contract": "all policies consume identical refined boxes and case inputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "cases", "variants")}, indent=2))


if __name__ == "__main__":
    main()
