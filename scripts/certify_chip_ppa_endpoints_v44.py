#!/usr/bin/env python3
"""Certify routed-PPA endpoints against immutable ORFS artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.ppa_external import parse_global_route_overflow
from scripts.run_chip_def_adapter_suite import flow_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppa-report",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_external_holdout_adapter_v44.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_holdout_inventory_v44.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_ppa_endpoint_certification_v44.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    reports = args.ppa_report or [
        ROOT / "outputs/reports/chip_external_ppa_holdout_v44_seed19.json"
    ]
    reports = [
        (path if path.is_absolute() else ROOT / path).resolve()
        for path in reports
    ]
    config = read(args.config)
    policy = config["primary_protocol"]
    manifest = read(args.manifest)
    cases = {
        row["design_key"]: row
        for row in manifest["inventory"]
    }
    certified = []
    missing = []
    for report_path in reports:
        report = read(report_path)
        for run in report.get("runs", []):
            if run.get("status") != "completed":
                missing.append(
                    {
                        "report": str(report_path.relative_to(ROOT)),
                        "case": run.get("case"),
                        "arm": run.get("arm"),
                        "seed": run.get("seed"),
                        "reason": f"run status: {run.get('status')}",
                    }
                )
                continue
            case = cases[run["case"]]
            nickname = case["design_nickname"]
            variant = flow_variant(policy, run["arm"], run["seed"])
            grt_log = (
                ROOT
                / "outputs/chip_ppa/orfs/logs"
                / case["platform"]
                / nickname
                / variant
                / "5_1_grt.log"
            )
            metrics_path = ROOT / run["metrics"]["metrics_path"]
            metric_values = read(metrics_path)
            overflow = parse_global_route_overflow(
                grt_log.read_text(errors="replace") if grt_log.is_file() else ""
            )
            endpoints = {
                "routed_wirelength": metric_values.get(
                    "detailedroute__route__wirelength"
                ),
                "global_route_overflow_total": overflow["total"],
                "global_route_overflow_max_h": overflow["max_horizontal"],
                "global_route_overflow_max_v": overflow["max_vertical"],
                "detailed_route_drc": metric_values.get(
                    "detailedroute__route__drc_errors"
                ),
                "post_route_wns_ns": metric_values.get(
                    "finish__timing__setup__ws"
                ),
                "post_route_tns_ns": metric_values.get(
                    "finish__timing__setup__tns"
                ),
                "post_route_hold_wns_ns": metric_values.get(
                    "finish__timing__hold__ws"
                ),
                "total_power": metric_values.get("finish__power__total"),
                "stdcell_area": metric_values.get(
                    "finish__design__instance__area__stdcell"
                ),
                "macro_area": metric_values.get(
                    "finish__design__instance__area__macros"
                ),
                "utilization": metric_values.get(
                    "finish__design__instance__utilization"
                ),
                "global_place_congestion": metric_values.get(
                    "globalplace__gpl__routability__congestion"
                ),
            }
            required = (
                "routed_wirelength",
                "global_route_overflow_total",
                "detailed_route_drc",
                "post_route_wns_ns",
                "post_route_tns_ns",
                "total_power",
                "stdcell_area",
                "macro_area",
            )
            absent = [name for name in required if endpoints[name] is None]
            row = {
                "report": str(report_path.relative_to(ROOT)),
                "case": run["case"],
                "seed": int(run["seed"]),
                "arm": run["arm"],
                "variant": variant,
                "endpoints": endpoints,
                "artifacts": {
                    "metrics": {
                        "path": str(metrics_path.relative_to(ROOT)),
                        "sha256": digest(metrics_path),
                    },
                    "global_route_log": {
                        "path": str(grt_log.relative_to(ROOT)),
                        "sha256": digest(grt_log),
                    },
                },
                "missing_required_endpoints": absent,
                "status": "certified" if not absent else "incomplete",
            }
            certified.append(row)
            if absent:
                missing.append(
                    {
                        "case": run["case"],
                        "arm": run["arm"],
                        "seed": run["seed"],
                        "missing": absent,
                    }
                )
    pairs = []
    by_key = {
        (row["report"], row["case"], row["seed"], row["arm"]): row
        for row in certified
    }
    for report_path in reports:
        report = read(report_path)
        for case in report.get("cases", []):
            for seed in report.get("seeds", []):
                candidate = by_key.get(
                    (str(report_path.relative_to(ROOT)), case, int(seed), "candidate")
                )
                control = by_key.get(
                    (str(report_path.relative_to(ROOT)), case, int(seed), "control")
                )
                if candidate is None or control is None:
                    continue
                deltas = {}
                for name in candidate["endpoints"]:
                    left = candidate["endpoints"][name]
                    right = control["endpoints"][name]
                    if left is None or right in (None, 0):
                        deltas[name] = None
                    else:
                        deltas[name] = (float(left) - float(right)) / float(right)
                pairs.append(
                    {
                        "case": case,
                        "seed": int(seed),
                        "candidate_path": candidate["artifacts"]["metrics"]["path"],
                        "control_path": control["artifacts"]["metrics"]["path"],
                        "relative_delta": deltas,
                    }
                )
    report = {
        "version": "2026-10-28-chip-ppa-endpoint-certification-v44",
        "status": "pass" if certified and not missing else "incomplete",
        "scope": "Raw ORFS METRICS_2.1.2 and GRT final-congestion artifacts",
        "reports": [str(path.relative_to(ROOT)) for path in reports],
        "runs_certified": len(certified),
        "pairs": pairs,
        "runs": certified,
        "missing": missing,
        "units": {
            "routed_wirelength": "database length units as emitted by ORFS METRICS_2.1.2",
            "global_route_overflow": "GRT congestion units",
            "wns_tns": "ns as emitted by ORFS METRICS_2.1.2",
            "power": "platform power units from ORFS METRICS_2.1.2",
            "area": "platform area units from ORFS METRICS_2.1.2",
        },
        "endpoint_policy": (
            "Global-route overflow is parsed from the final GRT report rather than "
            "substituted with global-place congestion. Timing comes from post-route "
            "setup WNS/TNS; no proxy is relabeled as STA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs_certified": len(certified),
                "pairs": len(pairs),
                "missing": len(missing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
