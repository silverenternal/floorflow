#!/usr/bin/env python3
"""Summarize the completed and failed v45 external holdout evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppa-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_partial.json",
    )
    parser.add_argument(
        "--sky-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_sky_pending.json",
    )
    parser.add_argument(
        "--endpoint-report",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_external_ppa_v45_partial_endpoint_certification.json",
    )
    parser.add_argument(
        "--adapter-report",
        type=Path,
        default=ROOT
        / "outputs/reports/chip_def_adapter_holdout_v45_seed19_legalizer_v3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_final.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_route_iteration(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = re.findall(
        r"Start (\d+)(?:st|nd|rd|th) optimization iteration",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(matches[-1]) if matches else None


def main() -> None:
    args = parse_args()
    ppa = load(args.ppa_report)
    sky = load(args.sky_report)
    endpoint = load(args.endpoint_report)
    adapter = load(args.adapter_report)
    coral_logs = {
        arm: ROOT
        / "outputs/chip_ppa/orfs/logs/asap7/coralnpu"
        / f"ppa_v45_{arm}_s19"
        / "5_2_route.tmp.log"
        for arm in ("candidate", "control")
    }
    sky_runs = {
        row["arm"]: {
            "status": row["status"],
            "exit_code": row["flow"]["exit_code"],
            "log": row["flow"]["log"],
        }
        for row in sky["runs"]
    }
    missing = [
        row["case"]
        for row in ppa["comparisons"]
        if row.get("status") != "completed"
    ]
    checks = {
        "adapter_certified": adapter["status"] == "pass",
        "completed_13_pairs": ppa["summary"]["completed_pairs"] == 13,
        "missing_two_cases": set(missing)
        == {"asap7/coralnpu", "sky130hd/microwatt"},
        "endpoint_certified_26_runs": endpoint["runs_certified"] == 26
        and not endpoint["missing"],
        "wins_ties_losses_preserved": ppa["wirelength_win_tie_loss"]
        == {"wins": 3, "ties": 4, "losses": 6},
        "full_holdout_complete": False,
    }
    report = {
        "version": "chip-external-ppa-v45-final-v1",
        "status": "reject_incomplete_external_holdout",
        "required_complete_pairs": 15,
        "completed_pairs": ppa["summary"]["completed_pairs"],
        "complete_designs": 13,
        "summary": ppa["summary"],
        "wirelength_win_tie_loss": ppa["wirelength_win_tie_loss"],
        "endpoint_certification": {
            "source": str(args.endpoint_report.relative_to(ROOT)),
            "runs_certified": endpoint["runs_certified"],
            "pairs": len(endpoint["pairs"]),
            "missing": endpoint["missing"],
        },
        "failed_cases": {
            "asap7/coralnpu": {
                "status": "detailed_route_timeout",
                "last_optimization_iteration": {
                    arm: last_route_iteration(path)
                    for arm, path in coral_logs.items()
                },
            },
            "sky130hd/microwatt": {
                "status": "global_route_congestion_failure",
                "runs": sky_runs,
            },
        },
        "checks": checks,
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                args.ppa_report,
                args.sky_report,
                args.endpoint_report,
                args.adapter_report,
            )
        },
        "claim_policy": (
            "No external routed-PPA improvement, subgroup SOTA, or 15-design "
            "generalization claim is allowed. The two failures remain in the "
            "denominator and cannot be replaced after outcomes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_pairs": report["completed_pairs"],
                "failed_cases": list(report["failed_cases"]),
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
