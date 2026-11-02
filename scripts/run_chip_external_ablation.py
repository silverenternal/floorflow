#!/usr/bin/env python3
"""Run and summarize the frozen external PPA mechanism ablation."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_chip_def_adapter_suite import case_manifest, load_json
from scripts.run_chip_ppa_subset import run_one

DEFAULT_ABLATION = ROOT / "configs/chip_external_ablation_v1.json"
DEFAULT_ADAPTER = ROOT / "configs/chip_external_adapter_v1.json"
DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_PPA = ROOT / "outputs/reports/chip_external_ppa_subset_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_external_ppa_ablation_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ppa", type=Path, default=DEFAULT_PPA)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--work-home", type=Path, default=ROOT / "outputs/chip_ppa/orfs")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def merged_case(
    manifest: dict[str, Any],
    adapter: dict[str, Any],
    design_key: str,
) -> dict[str, Any]:
    return {**case_manifest(manifest, design_key), **adapter["cases"][design_key]}


def get_completed_run(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    primary = load_json(args.ppa) if args.ppa.is_file() else {"runs": []}
    seed = int(args.config_data["seed"])
    for row in primary.get("runs", []):
        if (
            row["case"] == case["design_key"]
            and row["arm"] == "candidate"
            and int(row["seed"]) == seed
        ):
            return row
    return run_one(args, lock, case, "candidate", seed)


def summarize(rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_case = {row["case"]: row for row in full_rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        grouped.setdefault(row["case"], []).append(row)
    summaries = []
    for arm in ("dense", "no_structural", "no_hierarchy", "no_projector"):
        ratios = []
        drc = []
        power = []
        area = []
        for case, case_rows in grouped.items():
            match = next((row for row in case_rows if row["arm"] == arm), None)
            full = full_by_case.get(case)
            if match is None or full is None:
                continue
            candidate_wl = match["metrics_values"]["detailedroute__route__wirelength"]
            full_wl = full["metrics_values"]["detailedroute__route__wirelength"]
            ratios.append((candidate_wl - full_wl) / full_wl)
            drc.append(match["metrics_values"]["detailedroute__route__drc_errors"])
            power.append(
                (
                    match["metrics_values"]["finish__power__total"]
                    - full["metrics_values"]["finish__power__total"]
                )
                / full["metrics_values"]["finish__power__total"]
            )
            area.append(
                (
                    match["metrics_values"]["finish__design__instance__area__stdcell"]
                    - full["metrics_values"]["finish__design__instance__area__stdcell"]
                )
                / full["metrics_values"]["finish__design__instance__area__stdcell"]
            )
        summaries.append(
            {
                "arm": arm,
                "relative_wirelength_vs_full": sum(ratios) / len(ratios)
                if ratios
                else None,
                "drc_counts": drc,
                "power_relative_vs_full": power,
                "stdcell_area_relative_vs_full": area,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    adapter = load_json(args.adapter)
    args.ablation_data = load_json(args.ablation)
    args.config_data = {
        **adapter,
        "seed": int(args.ablation_data["seed"]),
    }
    lock = load_json(args.lock)
    manifest = load_json(args.manifest)
    cases = [
        merged_case(manifest, adapter, key)
        for key in args.ablation_data["case_selection"]["cases"]
    ]
    if args.cases:
        requested = set(args.cases)
        available = {case["design_key"] for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"cases are not in the frozen ablation: {missing}")
        cases = [case for case in cases if case["design_key"] in requested]
    for case in cases:
        for arm in args.ablation_data["arms"]:
            if arm == "full":
                continue
            placement = (
                ROOT
                / "outputs/reports/chip_def_adapter_ablation"
                / case["design_nickname"]
                / f"{arm}_seed19.json"
            )
            if not placement.is_file() or load_json(placement).get("status") != "pass":
                raise RuntimeError(f"missing passing ablation placement: {placement}")
    full_rows = [
        get_completed_run(args, lock, case, "full")
        for case in cases
    ]
    jobs = [
        (case, arm, int(args.config_data["seed"]))
        for case in cases
        for arm in ("dense", "no_structural", "no_hierarchy", "no_projector")
    ]
    rows = list(full_rows)
    if args.jobs <= 1:
        rows.extend(run_one(args, lock, case, arm, seed) for case, arm, seed in jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_one, args, lock, case, arm, seed): (
                    case,
                    arm,
                    seed,
                )
                for case, arm, seed in jobs
            }
            for future in as_completed(futures):
                rows.append(future.result())
    summaries = summarize(rows, full_rows)
    checks = {
        "all_arms_completed": all(
            sum(row["status"] == "completed" for row in rows if row["case"] == case)
            == 5
            for case in [item["design_key"] for item in cases]
        ),
        "all_drc_zero": all(
            row.get("metrics_values", {}).get("detailedroute__route__drc_errors") == 0
            for row in rows
            if row["status"] == "completed"
        ),
    }
    report = {
        "version": "2026-10-28-chip-external-ppa-ablation-v1",
        "status": "diagnostic_complete"
        if all(checks.values())
        else "diagnostic_incomplete",
        "scope": "Seed-19 external routed PPA mechanism ablation on three predeclared cases",
        "cases": args.ablation_data["case_selection"]["cases"],
        "arms": list(args.ablation_data["arms"]),
        "runs": sorted(rows, key=lambda row: (row["case"], row["arm"], row["seed"])),
        "summaries": summaries,
        "checks": checks,
        "claim_policy": args.ablation_data["reporting"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "diagnostic_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
