#!/usr/bin/env python3
"""Generate, legalize, and round-trip the external adapter audit subset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.ppa_external import parse_orfs_graph
from floorflow.chip_flow.refiner_config import append_refiner_protocol

DEFAULT_CONFIG = ROOT / "configs/chip_external_adapter_v1.json"
DEFAULT_LOCK = ROOT / "configs/chip_orfs_external_lock_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_WORK = ROOT / "outputs/chip_ppa/orfs"
ADAPTER_LEGALIZER_VERSION = "v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-home", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--seeds", type=int, nargs="+", default=[19])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore placement reports and ORFS PDN artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_def_adapter_cpu_v1.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_manifest(manifest: dict[str, Any], design_key: str) -> dict[str, Any]:
    for row in manifest["inventory"]:
        if row["design_key"] == design_key:
            return row
    raise ValueError(f"case missing from benchmark manifest: {design_key}")


def adapter_report_root(policy: dict[str, Any]) -> Path:
    tag = str(policy.get("protocol_tag", "")).strip()
    name = f"chip_def_adapter_{tag}" if tag else "chip_def_adapter"
    return ROOT / "outputs/reports" / name


def placement_root(policy: dict[str, Any]) -> Path:
    tag = str(policy.get("protocol_tag", "")).strip()
    name = f"placements_{tag}" if tag else "placements"
    return ROOT / "outputs/chip_ppa" / name


def case_slug(case: dict[str, Any]) -> str:
    """Use platform + nickname because nicknames repeat across platforms."""

    return f"{case['platform']}__{case['design_nickname']}"


def flow_variant(policy: dict[str, Any], arm: str, seed: int) -> str:
    tag = str(policy.get("protocol_tag", "")).strip()
    middle = f"{tag}_" if tag else ""
    return f"ppa_{middle}{arm}_s{seed}"


def adapter_flow_variant(
    case: dict[str, Any],
    arm: str,
    seed: int,
) -> str:
    """Keep ORFS artifacts unique when nicknames repeat or paths change."""

    return (
        f"def_adapter_{ADAPTER_LEGALIZER_VERSION}_{case_slug(case)}_{arm}_s{seed}"
    )


def coordinate_tolerance(case: dict[str, Any], policy: dict[str, Any]) -> float:
    return float(
        case.get(
            "coordinate_roundtrip_tolerance_um",
            policy["coordinate_roundtrip_tolerance_um"],
        )
    )


def run_placement(
    args: argparse.Namespace,
    design_key: str,
    case: dict[str, Any],
    arm: str,
    checkpoint: Path,
    seed: int,
    *,
    force_shelf: bool = False,
) -> dict[str, Any]:
    policy = args.config_data["primary_protocol"]
    output = adapter_report_root(policy) / case_slug(case) / f"{arm}_seed{seed}.json"
    if (
        getattr(args, "reuse", False)
        and not getattr(args, "force_recompute", False)
        and output.is_file()
    ):
        existing = load_json(output)
        if existing.get("status") == "pass":
            return {
                "design_key": design_key,
                "arm": arm,
                "seed": seed,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "report": str(output.relative_to(ROOT)),
                "reused": True,
            }
    command = [
        sys.executable,
        str(ROOT / "scripts/run_chip_external_placement.py"),
        "--case",
        case_slug(case),
        "--checkpoint",
        str(checkpoint),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--odb",
        str(ROOT / case["input_odb"]),
        "--device",
        args.device,
        "--nfe",
        str(args.config_data["primary_protocol"]["nfe"]),
        "--refiner-steps",
        str(args.config_data["primary_protocol"]["refiner_steps"]),
        "--projector-iterations",
        str(args.config_data["primary_protocol"]["candidate_projector_iterations"]),
        "--macro-halo-um",
        str(case["macro_halo_um"]),
        "--legalizer-relation",
        str(
            args.config_data["primary_protocol"].get(
                "legalizer_relation",
                "restored",
            )
        ),
        "--output",
        str(output),
    ]
    append_refiner_protocol(command, policy)
    if policy.get("protocol_tag"):
        command.extend(["--protocol-tag", str(policy["protocol_tag"])])
    if policy.get("instance_hierarchy"):
        command.append("--instance-hierarchy")
    if policy.get("preserve_reference_orientation"):
        command.append("--preserve-reference-orientation")
    if force_shelf:
        command.append("--force-shelf")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "design_key": design_key,
        "arm": arm,
        "seed": seed,
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": str(output.relative_to(ROOT)) if output.is_file() else None,
    }


def source_variant(design_key: str) -> str:
    if design_key == "nangate45/black_parrot":
        return "baseline"
    if design_key == "nangate45/tinyRocket":
        return "baseline"
    return "macro_count"


def prepare_variant(
    args: argparse.Namespace,
    case: dict[str, Any],
    variant: str,
) -> None:
    source = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / source_variant(case["design_key"])
    )
    destination = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / variant
    )
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("1_*"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    mem = source / "mem.json"
    if mem.is_file():
        shutil.copy2(mem, destination / mem.name)


def docker_floorplan(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    variant: str,
    tcl_path: Path,
) -> dict[str, Any]:
    config_path = f"/src/OpenROAD-flow-scripts/{case['config'].split('third_party/OpenROAD-flow-scripts/')[-1]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT / 'third_party/OpenROAD-flow-scripts'}:/src/OpenROAD-flow-scripts:ro",
        "-v",
        f"{ROOT}:/work",
        "-w",
        "/src/OpenROAD-flow-scripts/flow",
        "-e",
        "FLOW_HOME=/src/OpenROAD-flow-scripts/flow",
        "-e",
        "OPENROAD_EXE=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
        "-e",
        "YOSYS_EXE=/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys",
        "-e",
        "KLAYOUT_CMD=/usr/bin/klayout",
        "-e",
        "PYTHON_EXE=/usr/bin/python3",
        "-e",
        f"NUM_CORES={args.num_cores}",
        "-e",
        "REPORT_ENDPOINT_SLACK=1",
        "--entrypoint",
        "/bin/bash",
        lock["image"]["local_tag"],
        "-lc",
        " ".join(
            [
                "make",
                f"DESIGN_CONFIG={config_path}",
                f"FLOW_VARIANT={variant}",
                f"WORK_HOME=/work/{args.work_home.relative_to(ROOT)}",
                f"MACRO_PLACEMENT_TCL=/work/{tcl_path.relative_to(ROOT)}",
                "OR_SEED=1",
                "GRT_SEED=1",
                "floorplan",
            ]
        ),
    ]
    log_path = (
        ROOT
        / "outputs/reports/orfs"
        / case["platform"]
        / case["design_nickname"]
        / f"{variant}_floorplan.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "exit_code": result.returncode,
        "log": str(log_path.relative_to(ROOT)),
        "command": command,
    }


def export_placed_graph(
    args: argparse.Namespace,
    lock: dict[str, Any],
    odb: Path,
) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/work",
        "--entrypoint",
        "/bin/bash",
        lock["image"]["local_tag"],
        "-lc",
        (
            "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad "
            "-exit -no_init -no_splash -threads 1 "
            f"-db /work/{odb.relative_to(ROOT)} "
            "/work/scripts/orfs_export_macro_graph.tcl"
        ),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_orfs_graph(result.stdout)


def expected_macro_orientation(
    row: dict[str, Any],
    policy: str,
) -> str:
    if policy == "R0":
        return "R0"
    return str(row.get("orientation", "R0"))


def roundtrip_check(
    args: argparse.Namespace,
    lock: dict[str, Any],
    case: dict[str, Any],
    variant: str,
    boxes_path: Path,
    tolerance_um: float,
) -> dict[str, Any]:
    pdn = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / variant
        / "2_4_floorplan_pdn.odb"
    )
    if not pdn.is_file():
        return {"status": "fail", "reason": "PDN output is missing", "pdn": str(pdn)}
    graph = export_placed_graph(args, lock, pdn)
    expected = load_json(boxes_path)
    expected_by_name = {row["name"]: row for row in expected["macros"]}
    actual_by_name = {row["name"]: row for row in graph.macros}
    orientation_policy = str(
        args.config_data.get("mapping_contract", {}).get(
            "orientation",
            "adapter_report",
        )
    )
    names_match = set(expected_by_name) == set(actual_by_name)
    max_error = 0.0
    max_size_error = 0.0
    orientation_match = True
    placed = True
    for name, expected_row in expected_by_name.items():
        actual = actual_by_name.get(name)
        if actual is None:
            continue
        actual_x = actual["x_dbu"] / graph.dbu_per_micron
        actual_y = actual["y_dbu"] / graph.dbu_per_micron
        actual_width = actual["width_dbu"] / graph.dbu_per_micron
        actual_height = actual["height_dbu"] / graph.dbu_per_micron
        max_error = max(
            max_error,
            abs(actual_x - expected_row["x_um"]),
            abs(actual_y - expected_row["y_um"]),
        )
        max_size_error = max(
            max_size_error,
            abs(actual_width - expected_row["width_um"]),
            abs(actual_height - expected_row["height_um"]),
        )
        expected_orientation = expected_macro_orientation(
            expected_row,
            orientation_policy,
        )
        orientation_match &= actual["orientation"] == expected_orientation
        placed &= actual["placement_status"] == "LOCKED"
    checks = {
        "macro_names_match": names_match,
        "macro_count_matches": len(expected["macros"]) == len(graph.macros),
        "coordinates_within_site_pitch": max_error <= tolerance_um,
        "sizes_within_dbu_tolerance": max_size_error <= 0.001,
        "orientations_match": orientation_match,
        "expected_orientation_policy": orientation_policy,
        "all_macros_placed": placed,
        "pdn_completed": True,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "max_coordinate_error_um": max_error,
        "max_size_error_um": max_size_error,
        "coordinate_tolerance_um": tolerance_um,
        "pdn": str(pdn.relative_to(ROOT)),
    }


def run_roundtrip(
    args: argparse.Namespace,
    lock: dict[str, Any],
    design_key: str,
    case: dict[str, Any],
    arm: str,
    seed: int,
) -> dict[str, Any]:
    variant = adapter_flow_variant(case, arm, seed)
    policy = args.config_data["primary_protocol"]
    tcl_path = placement_root(policy) / case_slug(case) / f"{arm}_seed{seed}.tcl"
    boxes_path = tcl_path.with_suffix(".json")
    if not tcl_path.is_file():
        return {
            "design_key": design_key,
            "arm": arm,
            "seed": seed,
            "status": "fail",
            "reason": "placement TCl is missing",
        }
    prepare_variant(args, case, variant)
    pdn = (
        args.work_home
        / "results"
        / case["platform"]
        / case["design_nickname"]
        / variant
        / "2_4_floorplan_pdn.odb"
    )
    flow = (
        {
            "exit_code": 0,
            "log": None,
            "command": None,
            "reused": True,
        }
        if (
            args.reuse
            and not args.force_recompute
            and pdn.is_file()
        )
        else docker_floorplan(args, lock, case, variant, tcl_path)
    )
    result = {
        "design_key": design_key,
        "arm": arm,
        "seed": seed,
        "flow": flow,
    }
    if flow["exit_code"] == 0:
        result.update(
            roundtrip_check(
                args,
                lock,
                case,
                variant,
                boxes_path,
                coordinate_tolerance(
                    case,
                    args.config_data["hard_constraint_policy"],
                ),
            )
        )
    else:
        result.update({"status": "fail", "reason": "ORFS floorplan failed"})
    return result


def main() -> None:
    args = parse_args()
    args.config_data = load_json(args.config)
    lock = load_json(args.lock)
    manifest = load_json(args.manifest)
    placement_tasks = []
    for design_key in manifest["selection"]["selected_cases"]:
        case = case_manifest(manifest, design_key)
        adapter_case = args.config_data["cases"][design_key]
        case = {**case, **adapter_case}
        for arm in ("candidate", "control"):
            for seed in args.seeds:
                checkpoint = ROOT / args.config_data[arm]["seeds"][str(seed)]
                placement_tasks.append((design_key, case, arm, checkpoint, seed))

    def run_placement_task(task):
        design_key, case, arm, checkpoint, seed = task
        run = run_placement(args, design_key, case, arm, checkpoint, seed)
        if run["exit_code"] == 0 and run["report"]:
            placement_report = load_json(ROOT / run["report"])
            if placement_report["status"] != "pass":
                run["exit_code"] = 1
        return run

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            runs = list(executor.map(run_placement_task, placement_tasks))
    else:
        runs = [run_placement_task(task) for task in placement_tasks]
    roundtrip_tasks = []
    for design_key in manifest["selection"]["selected_cases"]:
        case = case_manifest(manifest, design_key)
        case = {**case, **args.config_data["cases"][design_key]}
        for arm in ("candidate", "control"):
            for seed in args.seeds:
                roundtrip_tasks.append((design_key, case, arm, seed))
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            roundtrips = list(
                executor.map(
                    lambda task: run_roundtrip(args, lock, *task),
                    roundtrip_tasks,
                )
            )
    else:
        roundtrips = [
            run_roundtrip(args, lock, *task)
            for task in roundtrip_tasks
        ]
    checks = {
        "placement_generation_passed": all(run["exit_code"] == 0 for run in runs),
        "orfs_roundtrips_passed": all(row["status"] == "pass" for row in roundtrips),
    }
    report = {
        "version": "2026-10-28-chip-def-adapter-suite-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "cases": manifest["selection"]["selected_cases"],
        "seeds": args.seeds,
        "device": args.device,
        "protocol": args.config_data["primary_protocol"],
        "placement_runs": runs,
        "roundtrips": roundtrips,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
