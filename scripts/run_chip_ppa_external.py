#!/usr/bin/env python3
"""Run one frozen ORFS case and audit Metrics_2.1.2 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.orfs_runtime import (
    DEFAULT_ORFS,
    docker_name,
    docker_prefix,
    flow_paths,
    load_json,
    resolve_paths,
    run_logged,
)

DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_WORK_HOME = ROOT / "outputs/chip_ppa/orfs"
DEFAULT_SMOKE = ROOT / "outputs/reports/chip_ppa_harness_smoke_v1.json"

PRIMARY_KEYS = (
    "detailedroute__route__wirelength",
    "detailedroute__route__drc_errors",
    "globalroute__timing__setup__ws",
    "globalroute__timing__setup__tns",
    "globalplace__gpl__routability__congestion",
    "finish__timing__setup__ws",
    "finish__timing__setup__tns",
    "finish__timing__hold__ws",
    "finish__timing__hold__tns",
    "finish__design__instance__area",
    "finish__design__instance__area__stdcell",
    "finish__design__instance__area__macros",
    "finish__design__instance__utilization",
    "finish__power__total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="black_parrot")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--orfs-root", type=Path, default=DEFAULT_ORFS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-home", type=Path, default=DEFAULT_WORK_HOME)
    parser.add_argument("--macro-placement-tcl", type=Path)
    parser.add_argument("--chipbench-flow", action="store_true")
    parser.add_argument("--chipbench-input-def", type=Path)
    parser.add_argument("--chipbench-def2db", action="store_true")
    parser.add_argument("--run-flow", action="store_true")
    parser.add_argument("--num-cores", type=int)
    parser.add_argument(
        "--flow-target",
        choices=("floorplan", "place", "finish", "evaluate_MacroPlace"),
        default="finish",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Stop after the requested flow target; useful for preflight only.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--skip-cts-repair-timing", action="store_true")
    parser.add_argument("--detailed-route-end-iter", type=int)
    parser.add_argument("--smoke-output", type=Path, default=DEFAULT_SMOKE)
    return parser.parse_args()


def case_record(manifest: dict[str, Any], case: str) -> dict[str, Any]:
    for row in manifest["inventory"]:
        design_key_name = row["design_key"].split("/")[-1]
        design_name = row.get("design_name", design_key_name)
        design_nickname = row.get("design_nickname", design_name)
        aliases = {
            row["case"],
            row["design_key"],
            f"{row['platform']}__{design_nickname}",
            f"{row['platform']}__{design_name}",
            f"{row['platform']}__{design_key_name}",
        }
        if case in aliases:
            return row
    raise ValueError(f"case is not in the frozen inventory: {case}")


def build_flow_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    marker = "third_party/OpenROAD-flow-scripts/"
    if marker in case["config"]:
        config = f"/src/OpenROAD-flow-scripts/{case['config'].split(marker)[-1]}"
        extra_make: list[str] = []
    else:
        config = f"/work/{case['config']}"
        design_home = Path(case["config"]).parent.parent.parent
        extra_make = [f"DESIGN_HOME=/work/{design_home}"]
    makefile = (
        ["-f", "/work/third_party/ChiPBench/flow/Makefile"]
        if getattr(args, "chipbench_flow", False)
        else []
    )
    flow_home = (
        ["FLOW_HOME=/work/third_party/ChiPBench/flow"]
        if getattr(args, "chipbench_flow", False)
        else []
    )
    targets = [args.flow_target]
    if (
        getattr(args, "chipbench_flow", False)
        and getattr(args, "chipbench_def2db", False)
        and args.flow_target == "finish"
    ):
        targets = [
            "do-2_4_floorplan_tapcell",
            "do-2_5_floorplan_pdn",
            "do-2_floorplan",
            "do-place",
            "do-cts",
            "do-route",
            "do-finish",
        ]
    make = [
        "make",
        *makefile,
        f"DESIGN_CONFIG={config}",
        *extra_make,
        *flow_home,
        f"FLOW_VARIANT={args.variant}",
        f"WORK_HOME=/work/{args.work_home.relative_to(ROOT)}",
        "OR_SEED=1",
        "GRT_SEED=1",
        *targets,
    ]
    route_end_iter = getattr(args, "detailed_route_end_iter", None)
    if route_end_iter is not None:
        make.append(
            "DETAILED_ROUTE_ARGS="
            f"'-droute_end_iter {int(route_end_iter)}'"
        )
    if (
        args.macro_placement_tcl is not None
        and not getattr(args, "chipbench_def2db", False)
    ):
        relative = args.macro_placement_tcl.resolve().relative_to(ROOT)
        make.insert(-1, f"MACRO_PLACEMENT_TCL=/work/{relative}")
    extra_env = (
        {"SKIP_CTS_REPAIR_TIMING": "1"}
        if getattr(args, "skip_cts_repair_timing", False)
        else None
    )
    return [
        *docker_prefix(
            lock,
            args.num_cores,
            extra_env,
            container_name=docker_name(args, "flow"),
        ),
        "-lc",
        " ".join(make),
    ]


def build_chipbench_def2db_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    if args.chipbench_input_def is None:
        raise ValueError("--chipbench-def2db requires --chipbench-input-def")
    config = f"/work/{case['config']}"
    design_home = Path(case["config"]).parent.parent.parent
    input_def = args.chipbench_input_def.resolve().relative_to(ROOT)
    make = [
        "make",
        "-f",
        "/work/third_party/ChiPBench/flow/Makefile",
        f"DESIGN_CONFIG={config}",
        f"DESIGN_HOME=/work/{design_home}",
        "FLOW_HOME=/work/third_party/ChiPBench/flow",
        f"FLOW_VARIANT={args.variant}",
        f"WORK_HOME=/work/{args.work_home.relative_to(ROOT)}",
        f"TARGET_DEF_PATH=/work/{input_def}",
        "TARGET_DB_PATH=2_3_floorplan_macro.odb",
        "def2db",
    ]
    return [
        *docker_prefix(
            lock,
            args.num_cores,
            container_name=docker_name(args, "def2db"),
        ),
        "-lc",
        " ".join(make),
    ]


def build_chipbench_pre_macro_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    results = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / args.variant
    )
    input_db = results / "2_3_floorplan_macro.odb"
    output_db = results / "2_3_floorplan_macro.firm.odb"
    script = ROOT / "scripts/chipbench_pre_macro_odb.tcl"
    command = " ".join(
        [
            f"INPUT_ODB=/work/{input_db.relative_to(ROOT)}",
            f"OUTPUT_ODB=/work/{output_db.relative_to(ROOT)}",
            "$OPENROAD_EXE -exit -no_init",
            f"/work/{script.relative_to(ROOT)}",
            "&&",
            f"mv /work/{output_db.relative_to(ROOT)}",
            f"/work/{input_db.relative_to(ROOT)}",
        ]
    )
    return [
        *docker_prefix(
            lock,
            args.num_cores,
            container_name=docker_name(args, "premacro"),
        ),
        "-lc",
        command,
    ]


def build_metrics_command(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    logs: Path,
    reports: Path,
    results: Path,
    metrics_path: Path,
) -> list[str]:
    def mounted(path: Path) -> str:
        return f"/work/{path.relative_to(ROOT)}"

    metrics_script = (
        "/work/third_party/ChiPBench/flow/util/genMetrics.py"
        if getattr(args, "chipbench_flow", False)
        else "util/genMetrics.py"
    )
    if getattr(args, "chipbench_flow", False):
        metric_args = [
            f"/usr/bin/python3 {metrics_script}",
            f"--design {case['design_nickname']}",
            f"--flowVariant {args.variant}",
            f"--platform {case['platform']}",
            f"--flowPath {mounted(args.work_home)}",
            f"--output {mounted(metrics_path)}",
        ]
    else:
        metric_args = [
            f"/usr/bin/python3 {metrics_script}",
            f"--design {case['design_name']}",
            f"--flowVariant {args.variant}",
            f"--platform {case['platform']}",
            f"--logs {mounted(logs)}",
            f"--reports {mounted(reports)}",
            f"--results {mounted(results)}",
            f"--output {mounted(metrics_path)}",
        ]
    return [
        *docker_prefix(
            lock,
            container_name=docker_name(args, "metrics"),
        ),
        "-lc",
        " ".join(metric_args),
    ]


def inspect_metrics(
    metrics: dict[str, Any],
    logs: Path,
    reports: Path,
    results: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    values = {key: metrics.get(key) for key in PRIMARY_KEYS}
    missing = [key for key, value in values.items() if value in (None, "N/A", "ERR")]
    route_json = logs / "5_2_route.json"
    grt_json = logs / "5_1_grt.json"
    finish_report = reports / "6_finish.rpt"
    endpoint_slack = reports / "6_final_endpoint_slack.json"
    route_db = results / "5_2_route.odb"
    final_db = results / "6_final.odb"
    artifacts = {
        "metrics_json": metrics_path.is_file() and metrics_path.stat().st_size > 0,
        "detailed_route_stage_json": route_json.is_file(),
        "global_route_stage_json": grt_json.is_file(),
        "detailed_route_odb": route_db.is_file(),
        "final_odb": final_db.is_file(),
        "finish_report": finish_report.is_file(),
        "endpoint_slack_json": endpoint_slack.is_file(),
    }
    drc = values["detailedroute__route__drc_errors"]
    checks = {
        "metrics_version_2_1_2": metrics.get("run__flow__metrics_version")
        == "Metrics_2.1.2",
        "primary_metrics_present": not missing,
        "detailed_route_terminated": values[
            "detailedroute__route__wirelength"
        ]
        not in (None, "N/A", "ERR"),
        "drc_is_zero": drc == 0,
        "all_required_artifacts_present": all(artifacts.values()),
    }
    return {
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "values": values,
        "missing_primary_metrics": missing,
        "artifacts": artifacts,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> None:
    args = resolve_paths(parse_args())
    lock = load_json(args.lock)
    manifest = load_json(args.manifest)
    case = case_record(manifest, args.case)
    logs, reports, results = flow_paths(args, case)
    metrics_path = (
        ROOT
        / "outputs/reports/orfs"
        / case["platform"]
        / case["design_nickname"]
        / args.variant
        / "metrics_2_1_2.json"
    )
    flow_run = None
    def2db_run = None
    pre_macro_run = None
    if args.run_flow:
        if args.chipbench_def2db:
            def2db_log = (
                ROOT
                / "outputs/reports/orfs"
                / case["platform"]
                / case["design_nickname"]
                / f"{args.variant}_def2db.log"
            )
            def2db_run = run_logged(
                build_chipbench_def2db_command(args, lock, case),
                def2db_log,
                args.timeout_seconds,
            )
            if def2db_run["exit_code"] != 0:
                raise RuntimeError(
                    f"ChiPBench def2db failed; see {def2db_log}"
                )
            pre_macro_log = (
                ROOT
                / "outputs/reports/orfs"
                / case["platform"]
                / case["design_nickname"]
                / f"{args.variant}_pre_macro.log"
            )
            pre_macro_run = run_logged(
                build_chipbench_pre_macro_command(args, lock, case),
                pre_macro_log,
                args.timeout_seconds,
            )
            if pre_macro_run["exit_code"] != 0:
                raise RuntimeError(
                    f"ChiPBench pre-macro ODB setup failed; "
                    f"see {pre_macro_log}"
                )
        command = build_flow_command(args, lock, case)
        flow_log = (
            ROOT
            / "outputs/reports/orfs"
            / case["platform"]
            / case["design_nickname"]
            / f"{args.variant}.log"
        )
        flow_run = run_logged(command, flow_log, args.timeout_seconds)
        if flow_run["exit_code"] != 0:
            report = {
                "version": "2026-10-28-chip-ppa-harness-smoke-v1",
                "status": "fail",
                "case": case["design_key"],
                "variant": args.variant,
                "flow_run": flow_run,
                "reason": "ORFS flow did not finish successfully",
            }
            args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            sys.exit(1)
    if args.skip_metrics:
        report = {
            "version": "2026-10-28-chip-ppa-harness-smoke-v1",
            "status": "preflight_pass",
            "case": case["design_key"],
            "variant": args.variant,
            "flow_target": args.flow_target,
            "flow_run": flow_run,
            "source": {
                "orfs_commit": lock["repository"]["commit"],
                "image_manifest_digest": lock["image"]["linux_amd64_manifest_digest"],
                "platform": case["platform"],
            },
            "checks": {
                "flow_requested": flow_run is not None,
                "flow_succeeded": bool(flow_run and flow_run["exit_code"] == 0),
                "image_pinned": lock["image"]["linux_amd64_manifest_digest"].startswith(
                    "sha256:"
                ),
            },
            "policy": {
                "candidate_evaluated": False,
                "metrics_skipped_by_request": True,
            },
        }
        report["status"] = (
            "preflight_pass"
            if all(report["checks"].values())
            else "fail"
        )
        args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
        args.smoke_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
        if report["status"] != "preflight_pass":
            sys.exit(1)
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_run = run_logged(
        build_metrics_command(
            args,
            lock,
            case,
            logs,
            reports,
            results,
            metrics_path,
        ),
        metrics_path.with_suffix(".log"),
        timeout=300,
    )
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    metric_audit = inspect_metrics(metrics, logs, reports, results, metrics_path)
    checks = {
        "frozen_case": case["design_key"] in manifest["selection"]["selected_cases"],
        "image_pinned": lock["image"]["linux_amd64_manifest_digest"].startswith("sha256:"),
        "metrics_command_succeeded": metrics_run["exit_code"] == 0,
        "metrics_audit": metric_audit["status"] == "pass",
    }
    report = {
        "version": "2026-10-28-chip-ppa-harness-smoke-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "case": case["design_key"],
        "variant": args.variant,
        "macro_placement_tcl": (
            str(args.macro_placement_tcl)
            if args.macro_placement_tcl is not None
            else "design default"
        ),
        "source": {
            "orfs_commit": lock["repository"]["commit"],
            "image_reference": lock["image"]["mirror_reference"],
            "image_manifest_digest": lock["image"]["linux_amd64_manifest_digest"],
            "image_config_digest": lock["image"]["linux_amd64_config_digest"],
            "local_tag": lock["image"]["local_tag"],
            "platform": case["platform"],
        },
        "flow_settings": {
            **lock["flow_settings"],
            "flow_variant": args.variant,
            "or_seed": 1,
            "grt_seed": 1,
        },
        "flow_run": flow_run,
        "def2db_run": def2db_run,
        "pre_macro_run": pre_macro_run,
        "metrics_run": metrics_run,
        "metric_audit": metric_audit,
        "checks": checks,
        "policy": {
            "candidate_evaluated": False,
            "surrogate_used_for_missing_metric": False,
            "all_primary_metrics_from_orfs": True,
        },
    }
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
