#!/usr/bin/env python3
"""Run the frozen candidate/control subset through routed ORFS PPA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_chip_def_adapter_suite import (
    adapter_report_root,
    case_manifest,
    case_slug,
    coordinate_tolerance,
    flow_variant,
    load_json,
    placement_root,
    prepare_variant,
    roundtrip_check,
    run_placement,
)
from scripts.run_chip_ppa_external import (
    build_metrics_command,
    docker_prefix,
    flow_paths,
    inspect_metrics,
    run_logged,
)

DEFAULT_CONFIG = ROOT / "configs/chip_external_adapter_v1.json"
DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_WORK = ROOT / "outputs/chip_ppa/orfs"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_external_ppa_subset_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-home", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[19, 20, 21])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=32)
    parser.add_argument("--macro-halo-override", type=float)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def merged_case(
    manifest: dict[str, Any],
    adapter_config: dict[str, Any],
    design_key: str,
) -> dict[str, Any]:
    case = case_manifest(manifest, design_key)
    return {**case, **adapter_config["cases"][design_key]}


def make_flow_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    arm: str,
    seed: int,
    tcl_path: Path,
) -> list[str]:
    variant = flow_variant(args.config_data["primary_protocol"], arm, seed)
    config_path = (
        "/src/OpenROAD-flow-scripts/"
        + case["config"].split("third_party/OpenROAD-flow-scripts/")[-1]
    )
    make = [
        "make",
        f"DESIGN_CONFIG={config_path}",
        f"FLOW_VARIANT={variant}",
        f"WORK_HOME=/work/{args.work_home.relative_to(ROOT)}",
        f"MACRO_PLACEMENT_TCL=/work/{tcl_path.relative_to(ROOT)}",
        "OR_SEED=1",
        "GRT_SEED=1",
        "finish",
    ]
    return [*docker_prefix(lock, args.num_cores), "-lc", " ".join(make)]


def metrics_path(
    case: dict[str, Any],
    arm: str,
    seed: int,
    policy: dict[str, Any],
) -> Path:
    tag = f"{policy['protocol_tag']}_" if policy.get("protocol_tag") else ""
    return (
        ROOT
        / "outputs/reports/orfs/ppa"
        / case["platform"]
        / case["design_nickname"]
        / f"{tag}{arm}_seed{seed}_metrics_2_1_2.json"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def placement_report_path(
    case: dict[str, Any],
    arm: str,
    seed: int,
    policy: dict[str, Any],
) -> Path:
    return adapter_report_root(policy) / case_slug(case) / f"{arm}_seed{seed}.json"


def run_one(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    arm: str,
    seed: int,
) -> dict[str, Any]:
    policy = args.config_data["primary_protocol"]
    variant = flow_variant(policy, arm, seed)
    prepare_variant(args, case, variant)
    tcl_path = (
        placement_root(policy) / case_slug(case) / f"{arm}_seed{seed}.tcl"
    )
    logs, reports, results = flow_paths(
        argparse.Namespace(**{**vars(args), "variant": variant}),
        case,
    )
    metric_path = metrics_path(case, arm, seed, policy)
    final_db = results / "6_final.odb"
    flow_result = None
    if not final_db.is_file():
        flow_log = (
            ROOT
            / "outputs/reports/orfs/ppa"
            / case["platform"]
            / case["design_nickname"]
        / f"{policy.get('protocol_tag', '')}_{arm}_seed{seed}.log".lstrip("_")
        )
        flow_result = run_logged(
            make_flow_command(args, lock, case, arm, seed, tcl_path),
            flow_log,
            args.timeout_seconds,
        )
        if flow_result["exit_code"] != 0:
            return {
                "case": case["design_key"],
                "source_family": case.get("source_family"),
                "arm": arm,
                "seed": seed,
                "status": "failed",
                "flow": flow_result,
            }
    if not metric_path.is_file():
        metrics_args = argparse.Namespace(**{**vars(args), "variant": variant})
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_result = run_logged(
            build_metrics_command(
                metrics_args,
                lock,
                case,
                logs,
                reports,
                results,
                metric_path,
            ),
            metric_path.with_suffix(".log"),
            timeout=300,
        )
        if metrics_result["exit_code"] != 0:
            return {
                "case": case["design_key"],
                "source_family": case.get("source_family"),
                "arm": arm,
                "seed": seed,
                "status": "failed",
                "flow": flow_result,
                "metrics_run": metrics_result,
            }
    metrics = load_json(metric_path)
    audit = inspect_metrics(metrics, logs, reports, results, metric_path)
    placement_boxes = (
        ROOT
        / "outputs/chip_ppa"
        / (
            f"placements_{policy['protocol_tag']}"
            if policy.get("protocol_tag")
            else "placements"
        )
        / case_slug(case)
        / f"{arm}_seed{seed}.json"
    )
    roundtrip = roundtrip_check(
        args,
        lock,
        case,
        variant,
        placement_boxes,
        coordinate_tolerance(
            case,
            args.config_data["hard_constraint_policy"],
        ),
    )
    return {
        "case": case["design_key"],
        "source_family": case.get("source_family"),
        "arm": arm,
        "seed": seed,
        "status": (
            "completed"
            if audit["status"] == "pass" and roundtrip["status"] == "pass"
            else "metric_failure"
        ),
        "variant": variant,
        "macro_placement_tcl": str(tcl_path.relative_to(ROOT)),
        "macro_placement_tcl_sha256": sha256(tcl_path),
        "flow": flow_result,
        "metrics": audit,
        "metrics_values": audit["values"],
        "roundtrip": roundtrip,
    }


def aggregate(
    rows: list[dict[str, Any]],
    seeds: list[int],
    expected_cases: list[str] | None = None,
) -> dict[str, Any]:
    comparisons = []
    by_key = {(row["case"], row["arm"], row["seed"]): row for row in rows}
    row_cases = {row["case"] for row in rows}
    cases = sorted(set(expected_cases) if expected_cases is not None else row_cases)
    for case in cases:
        for seed in seeds:
            candidate = by_key.get((case, "candidate", seed))
            control = by_key.get((case, "control", seed))
            if not candidate or not control:
                comparisons.append({"case": case, "seed": seed, "status": "missing"})
                continue
            if candidate["status"] != "completed" or control["status"] != "completed":
                comparisons.append({"case": case, "seed": seed, "status": "failed"})
                continue
            candidate_values = candidate["metrics_values"]
            control_values = control["metrics_values"]
            row = {
                "case": case,
                "source_family": candidate.get("source_family"),
                "seed": seed,
                "status": "completed",
            }
            for metric in (
                "detailedroute__route__wirelength",
                "globalplace__gpl__routability__congestion",
                "finish__timing__setup__ws",
                "finish__power__total",
                "finish__design__instance__area__stdcell",
            ):
                left = candidate_values.get(metric)
                right = control_values.get(metric)
                row[metric] = {
                    "candidate": left,
                    "control": right,
                    "relative": (left - right) / right if right not in (None, 0) else None,
                }
            row["candidate_drc"] = candidate_values.get("detailedroute__route__drc_errors")
            row["control_drc"] = control_values.get("detailedroute__route__drc_errors")
            comparisons.append(row)
    completed = [row for row in comparisons if row["status"] == "completed"]
    wire = [row["detailedroute__route__wirelength"]["relative"] for row in completed]
    power = [row["finish__power__total"]["relative"] for row in completed]
    area = [row["finish__design__instance__area__stdcell"]["relative"] for row in completed]
    congestion = [
        row["globalplace__gpl__routability__congestion"]["relative"]
        for row in completed
        if row["globalplace__gpl__routability__congestion"]["relative"] is not None
    ]
    timing = [
        row["finish__timing__setup__ws"]["candidate"] - row["finish__timing__setup__ws"]["control"]
        for row in completed
    ]
    seed_means = {}
    for seed in seeds:
        values = [
            row["detailedroute__route__wirelength"]["relative"]
            for row in completed
            if row["seed"] == seed
        ]
        seed_means[str(seed)] = sum(values) / len(values) if values else None
    case_means = {}
    for case in cases:
        values = [
            row["detailedroute__route__wirelength"]["relative"]
            for row in completed
            if row["case"] == case
        ]
        case_means[case] = sum(values) / len(values) if values else None
    family_means = {}
    for family in sorted(
        {row["source_family"] for row in completed if row["source_family"]}
    ):
        values = [
            row["detailedroute__route__wirelength"]["relative"]
            for row in completed
            if row["source_family"] == family
        ]
        family_means[family] = sum(values) / len(values) if values else None

    def without_group(
        group_key: str,
        values_to_exclude: list[Any],
    ) -> dict[str, float]:
        result = {}
        for excluded in values_to_exclude:
            selected = [
                row["detailedroute__route__wirelength"]["relative"]
                for row in completed
                if row[group_key] != excluded
            ]
            result[str(excluded)] = sum(selected) / len(selected) if selected else None
        return result

    def without_family(values_to_exclude: list[Any]) -> dict[str, float]:
        result = {}
        for excluded in values_to_exclude:
            selected = [
                row["detailedroute__route__wirelength"]["relative"]
                for row in completed
                if row["source_family"] != excluded
            ]
            result[str(excluded)] = sum(selected) / len(selected) if selected else None
        return result

    win_counts = {
        "wins": sum(value < -1.0e-12 for value in wire),
        "ties": sum(abs(value) <= 1.0e-12 for value in wire),
        "losses": sum(value > 1.0e-12 for value in wire),
    }
    all_drc_zero = all(
        row["candidate_drc"] == 0 and row["control_drc"] == 0 for row in completed
    )
    congestion_nonregression = bool(congestion) and (
        sum(congestion) / len(congestion) <= 0.02
    )
    timing_nonregression = bool(timing) and (
        sum(timing) / len(timing) >= -0.02
    )
    checks = {
        "all_pairs_completed": len(completed) == len(cases) * len(seeds),
        "primary_mean_at_least_2pct_better": bool(wire) and sum(wire) / len(wire) <= -0.02,
        "drc_zero_both_arms": all_drc_zero,
        "power_within_5pct": bool(power) and max(power) <= 0.05,
        "stdcell_area_within_5pct": bool(area) and max(area) <= 0.05,
        "congestion_mean_at_most_2pct": congestion_nonregression,
        "timing_ws_mean_nonregression_20ps": timing_nonregression,
        "congestion_or_timing_non_regression": (
            congestion_nonregression or timing_nonregression
        ),
    }
    gate_keys = (
        "all_pairs_completed",
        "primary_mean_at_least_2pct_better",
        "drc_zero_both_arms",
        "power_within_5pct",
        "stdcell_area_within_5pct",
        "congestion_or_timing_non_regression",
    )
    promotion = all(checks[key] for key in gate_keys)
    return {
        "comparisons": comparisons,
        "seed_relative_wirelength_means": seed_means,
        "case_relative_wirelength_means": case_means,
        "family_relative_wirelength_means": family_means,
        "leave_one_family_out": without_family(sorted(family_means)),
        "leave_one_case_out": without_group("case", cases),
        "leave_one_seed_out": without_group("seed", seeds),
        "wirelength_win_tie_loss": win_counts,
        "summary": {
            "primary_relative_wirelength_mean": sum(wire) / len(wire) if wire else None,
            "primary_relative_wirelength_range": (
                [min(wire), max(wire)] if wire else None
            ),
            "congestion_relative_mean": sum(congestion) / len(congestion) if congestion else None,
            "timing_ws_delta_mean": sum(timing) / len(timing) if timing else None,
            "power_relative_max": max(power) if power else None,
            "stdcell_area_relative_max": max(area) if area else None,
            "completed_pairs": len(completed),
        },
        "checks": checks,
        "promotion_gate": "pass" if promotion else "reject_ppa_claim",
    }


def main() -> None:
    args = parse_args()
    adapter_config = load_json(args.config)
    args.config_data = adapter_config
    lock = load_json(args.lock)
    manifest = load_json(args.manifest)
    cases = [
        merged_case(manifest, adapter_config, key)
        for key in manifest["selection"]["selected_cases"]
    ]
    if args.macro_halo_override is not None:
        for case in cases:
            case["macro_halo_um"] = float(args.macro_halo_override)
    if args.cases:
        requested = set(args.cases)
        available = {case["design_key"] for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"cases are not in the frozen selection: {missing}")
        cases = [case for case in cases if case["design_key"] in requested]
    common_route_resolutions = []
    if not args.collect_only:
        for case in cases:
            for arm in ("candidate", "control"):
                for seed in args.seeds:
                    checkpoint = ROOT / adapter_config[arm]["seeds"][str(seed)]
                    placement_report = placement_report_path(
                        case,
                        arm,
                        seed,
                        adapter_config["primary_protocol"],
                    )
                    if placement_report.is_file():
                        placement = load_json(placement_report)
                        if placement.get("status") == "pass":
                            continue
                    result = run_placement(
                        args,
                        case["design_key"],
                        case,
                        arm,
                        checkpoint,
                        seed,
                    )
                    if result["exit_code"] != 0:
                        raise RuntimeError(f"placement generation failed: {result}")
    if (
        not args.collect_only
        and adapter_config["primary_protocol"].get("pairwise_common_route")
    ):
        for case in cases:
            for seed in args.seeds:
                reports = {
                    arm: load_json(
                        placement_report_path(
                            case,
                            arm,
                            seed,
                            adapter_config["primary_protocol"],
                        )
                    )
                    for arm in ("candidate", "control")
                }
                routes = {
                    arm: report.get("legalizer_route")
                    for arm, report in reports.items()
                }
                if routes["candidate"] == routes["control"]:
                    continue
                for arm in ("candidate", "control"):
                    checkpoint = ROOT / adapter_config[arm]["seeds"][str(seed)]
                    result = run_placement(
                        args,
                        case["design_key"],
                        case,
                        arm,
                        checkpoint,
                        seed,
                        force_shelf=True,
                    )
                    if result["exit_code"] != 0:
                        raise RuntimeError(
                            f"pairwise common-route regeneration failed: {result}"
                        )
                common_route_resolutions.append(
                    {
                        "case": case["design_key"],
                        "seed": seed,
                        "before": routes,
                        "after": "shelf_pack_forced",
                    }
                )
    jobs = [
        (case, arm, seed)
        for case in cases
        for arm in ("candidate", "control")
        for seed in args.seeds
    ]
    if args.collect_only:
        jobs = [
            (case, arm, seed)
            for case, arm, seed in jobs
            if (
                args.work_home
                / "results"
                / case["platform"]
                / case["design_nickname"]
                / flow_variant(
                    adapter_config["primary_protocol"],
                    arm,
                    seed,
                )
                / "6_final.odb"
            ).is_file()
        ]
    rows = []
    if args.jobs <= 1:
        rows = [run_one(args, lock, case, arm, seed) for case, arm, seed in jobs]
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
    summary = aggregate(
        rows,
        args.seeds,
        [case["design_key"] for case in cases],
    )
    report = {
        "version": "2026-10-28-chip-external-ppa-subset-v1",
        "status": summary["promotion_gate"],
        "cases": [case["design_key"] for case in cases],
        "seeds": args.seeds,
        "protocol": {
            "primary": adapter_config["primary_protocol"],
            "orfs_image": lock["image"]["local_tag"],
            "num_cores": args.num_cores,
            "or_seed": 1,
            "grt_seed": 1,
            "pairwise_common_route_resolutions": common_route_resolutions,
            "macro_halo_override": args.macro_halo_override,
        },
        "runs": sorted(rows, key=lambda row: (row["case"], row["arm"], row["seed"])),
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": summary["summary"]}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
