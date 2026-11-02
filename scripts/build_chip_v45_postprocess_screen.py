#!/usr/bin/env python3
"""Evaluate the frozen multi-case v2 postprocess screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(candidate: float, control: float) -> float | None:
    return candidate / control - 1.0 if control != 0.0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_v45_postprocess_v2_screen.json",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_holdout_v44_all_seed19.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_postprocess_screen.json",
    )
    return parser.parse_args()


def baseline_metrics(
    reports: list[dict[str, Any]],
    case: str,
) -> dict[str, float]:
    for report in reports:
        for run in report.get("runs", []):
            if run.get("case") == case and run.get("arm") == "candidate":
                values = run.get("metrics", {}).get("values")
                if values:
                    return values
    raise ValueError(f"candidate baseline missing for {case}")


def metric_comparisons(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    keys = {
        "routed_wirelength": "detailedroute__route__wirelength",
        "congestion": "globalplace__gpl__routability__congestion",
        "setup_ws": "finish__timing__setup__ws",
        "hold_ws": "finish__timing__hold__ws",
        "power": "finish__power__total",
        "instance_area": "finish__design__instance__area",
        "drc_errors": "detailedroute__route__drc_errors",
    }
    return {
        name: {
            "before": before.get(key),
            "after": after.get(key),
            "relative": (
                relative(float(after[key]), float(before[key]))
                if before.get(key) is not None and after.get(key) is not None
                else None
            ),
        }
        for name, key in keys.items()
    }


def case_checks(metrics: dict[str, dict[str, Any]], placement: dict[str, Any]) -> dict[str, bool]:
    return {
        "proxy_improves": placement["proxy_relative"] < 0.0,
        "legal_after": bool(placement["legal_after"]),
        "wirelength_improves": metrics["routed_wirelength"]["relative"] < 0.0,
        "congestion_nonregression": metrics["congestion"]["relative"] <= 0.0,
        "setup_ws_nonregression": (
            metrics["setup_ws"]["after"] >= metrics["setup_ws"]["before"]
        ),
        "drc_zero": metrics["drc_errors"]["after"] == 0,
        "power_at_most_5pct_regression": metrics["power"]["relative"] <= 0.05,
        "area_at_most_5pct_regression": (
            metrics["instance_area"]["relative"] <= 0.05
        ),
    }


def main() -> None:
    args = parse_args()
    config = load(args.config)
    baseline_reports = [
        load(ROOT / "outputs/reports/chip_external_ppa_holdout_v44_seed19.json"),
        load(args.baseline),
    ]
    cases = []
    for case in config["selection"]["cases"]:
        artifacts = config["artifact_map"][case]
        placement = load(ROOT / artifacts["placement_report"])
        after = load(ROOT / artifacts["routed_report"])["metric_audit"]["values"]
        metrics = metric_comparisons(baseline_metrics(baseline_reports, case), after)
        checks = case_checks(metrics, placement)
        cases.append(
            {
                "case": case,
                "status": "pass" if all(checks.values()) else "reject",
                "proxy_relative": placement["proxy_relative"],
                "displacement_um": placement["displacement_um"],
                "metrics": metrics,
                "checks": checks,
            }
        )
    means = {
        name: float(np.mean([row["metrics"][name]["relative"] for row in cases]))
        for name in ("routed_wirelength", "congestion", "setup_ws", "power")
    }
    report = {
        "version": "2026-10-29-chip-v45-postprocess-screen-v1",
        "status": (
            "pass_multicase_screen"
            if all(row["status"] == "pass" for row in cases)
            else "rejected_multicase"
        ),
        "cases": cases,
        "mean_relative": means,
        "decision": (
            "The fixed proxy-only displacement recipe is rejected multicase. "
            "Do not apply it to the frozen v45 routed holdout as a claimed "
            "improvement."
        ),
        "claim_boundary": (
            "The single black_parrot smoke remains a positive diagnostic only. "
            "It cannot override the frozen multicase rejection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
