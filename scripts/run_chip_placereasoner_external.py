#!/usr/bin/env python3
"""Run PlanB macro coordinates through PlaceReasoner's fixed routed flow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CONTRACT_MODULE_PATH = ROOT / "src/floorflow/chip_flow/placereasoner_bench.py"
SPEC = importlib.util.spec_from_file_location("placereasoner_bench", CONTRACT_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load benchmark contract module: {CONTRACT_MODULE_PATH}")
CONTRACT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACT_MODULE
SPEC.loader.exec_module(CONTRACT_MODULE)
PlaceReasonerTask = CONTRACT_MODULE.PlaceReasonerTask
benchmark_tasks = CONTRACT_MODULE.benchmark_tasks
load_benchmark_contract = CONTRACT_MODULE.load_benchmark_contract
normalize_coordinates = CONTRACT_MODULE.normalize_coordinates
sha256_file = CONTRACT_MODULE.sha256_file
validate_benchmark_root = CONTRACT_MODULE.validate_benchmark_root
validate_coordinate_count = CONTRACT_MODULE.validate_coordinate_count

DEFAULT_CONTRACT = ROOT / "configs/chip_placereasoner_bench_v1.json"
DEFAULT_REPO = ROOT / "third_party/PlaceReasoner-Beta"
DEFAULT_BENCH = DEFAULT_REPO / "Benchmark/PlaceReasoner-Bench"

PLANB_SAFE_PLACE_PROC = r"""
proc safe_place {name x y {orient R0}} {
    set inst [[ord::get_db_block] findInst $name]
    if {$inst eq "NULL" || $inst eq ""} {
        puts "SKIP: macro '$name' not found"
        return
    }
    $inst setPlacementStatus "NONE"
    set tech [ord::get_db_tech]
    set dbu  [$tech getDbUnitsPerMicron]
    set g 1
    if {[$tech hasManufacturingGrid]} { set g [$tech getManufacturingGrid] }
    set xdbu [expr {round(round($x * $dbu / double($g)) * $g)}]
    set ydbu [expr {round(round($y * $dbu / double($g)) * $g)}]
    $inst setLocation $xdbu $ydbu
    $inst setOrient $orient
    $inst setPlacementStatus "PLACED"
    puts "OK: $name placed at ($xdbu, $ydbu) orient $orient"
}
puts "planb_safe_place proc loaded"
"""


class NoVLM:
    """Fail if a deterministic adapter ever attempts an LLM call."""

    def chat(self, *_args, **_kwargs):
        raise RuntimeError("this adapter never permits VLM calls")


def install_external_place_proc(orchestrator) -> str:
    """Install a restart-safe OpenDB placement proc for external macro names."""

    init_script = Path(orchestrator.ops.scripts["init"])
    text = init_script.read_text(encoding="utf-8")
    marker = "planb_safe_place proc loaded"
    if marker not in text:
        init_script.write_text(
            text.rstrip() + "\n\n" + PLANB_SAFE_PLACE_PROC + "\n",
            encoding="utf-8",
        )
    orchestrator.ops.s.send_command(PLANB_SAFE_PLACE_PROC, timeout=30)
    return str(init_script)


def disable_image_rendering(orchestrator) -> None:
    """Bypass GUI/Qt snapshots while preserving registry metadata."""

    session = orchestrator.ops.s
    session.save_image = lambda *_args, **_kwargs: "[disabled]"

    def register_without_rendering(
        stage: str,
        name: str,
        desc: str = "",
        timeout: float = 30.0,
        **meta,
    ):
        path = orchestrator.ctx.img(name, stage=stage)
        return orchestrator.ctx.registry.register(
            stage,
            str(path),
            desc,
            **meta,
        )

    orchestrator.ctx.snapshot = register_without_rendering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--task", required=True, help="design/aspect, e.g. bp_fe/ar1")
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="output root; required unless --dry-run",
    )
    parser.add_argument(
        "--stop-after",
        choices=("init", "detail", "route"),
        default="route",
    )
    parser.add_argument("--route-mode", choices=("light", "timing"), default="timing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def find_task(tasks: tuple[PlaceReasonerTask, ...], task_id: str) -> PlaceReasonerTask:
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown benchmark task: {task_id}")


def write_report(path: Path | None, payload: dict) -> None:
    text = json.dumps(payload, indent=2, default=str) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(text, end="")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    contract = load_benchmark_contract(args.contract)
    tasks = benchmark_tasks(contract)
    task = find_task(tasks, args.task)
    errors = validate_benchmark_root(args.benchmark_root, tasks)
    if errors:
        raise ValueError("benchmark inventory is incomplete: " + "; ".join(errors[:5]))
    coordinates = normalize_coordinates(json.loads(args.coords.read_text(encoding="utf-8")))
    validate_coordinate_count(coordinates, task.macro_count)
    design_dir = args.benchmark_root / task.design
    files = {
        "verilog": design_dir / f"{task.design}.v",
        "sdc": design_dir / "constraint.sdc",
        "floorplan_def": design_dir / task.floorplan_def,
    }
    base = {
        "version": "chip-placereasoner-run-v1",
        "task": task.task_id,
        "design": task.design,
        "top": task.top,
        "aspect_id": task.aspect_id,
        "macro_count": task.macro_count,
        "repo_commit": contract["repo_commit"],
        "dataset_id": contract["dataset_id"],
        "coordinates_sha256": sha256_file(args.coords),
        "input_sha256": {name: sha256_file(path) for name, path in files.items()},
        "flow_contract": {
            "route_mode": args.route_mode,
            "stop_after": args.stop_after,
            "vlm": False,
            "legalizer": "PlaceReasoner fixed detail_place/CTS/route templates",
        },
    }
    if args.dry_run:
        write_report(
            args.report,
            {
                **base,
                "status": "dry_run",
                "openroad_on_path": shutil.which("openroad") is not None,
                "next_command": [
                    "python",
                    str(Path(__file__).resolve()),
                    "--task",
                    task.task_id,
                    "--coords",
                    str(args.coords.resolve()),
                    "--work-dir",
                    "<output-root>",
                    "--stop-after",
                    args.stop_after,
                    "--route-mode",
                    args.route_mode,
                    "--report",
                    "<report.json>",
                ],
            },
        )
        return
    if args.work_dir is None:
        raise ValueError("--work-dir is required unless --dry-run")
    if shutil.which("openroad") is None:
        raise RuntimeError(
            "openroad is not on PATH. Run this adapter inside a pinned OpenROAD "
            "container; the adapter intentionally refuses simulation mode."
        )

    sys.path.insert(0, str(args.repo_root.resolve()))
    from macroflow.config import FlowConfig
    from macroflow.orchestrator import MacroFlowOrchestrator

    config = FlowConfig(
        data_dir=design_dir,
        design=task.design,
        top=task.top,
        model_config=str(args.repo_root / "config.yaml"),
        work_dir=args.work_dir,
        route_mode=args.route_mode,
        floorplan_def=task.floorplan_def,
        tech_dir=args.repo_root / "Digital_lib/Nangate45",
    )
    orchestrator = MacroFlowOrchestrator(
        config,
        max_refine=0,
        gate_a_max_refine=0,
        macro_candidates=1,
        parallel_workers=1,
        post_route_opt_loops=0,
    )
    report = {**base, "status": "failed_before_init", "stages": {}}
    try:
        orchestrator.setup(vlm=NoVLM())
        disable_image_rendering(orchestrator)
        init_result = orchestrator.ops.init_design()
        report["external_place_proc_script"] = install_external_place_proc(
            orchestrator
        )
        report["stages"]["init"] = init_result
        if args.stop_after == "init":
            report["status"] = "init_complete"
            write_report(args.report, report)
            return
        expected_names = set(orchestrator.ops.st.macro_sizes)
        coordinate_names = {row["name"] for row in coordinates}
        if coordinate_names != expected_names:
            raise ValueError(
                "coordinate names do not match the loaded design: "
                f"missing={sorted(expected_names - coordinate_names)[:8]}, "
                f"extra={sorted(coordinate_names - expected_names)[:8]}"
            )
        detail = orchestrator.ops.detail_place(entries=coordinates, tag="planb")
        report["stages"]["detail"] = detail
        if detail.get("stage") != "detailed":
            report["status"] = "detail_failed"
            write_report(args.report, report)
            raise SystemExit(1)
        if args.stop_after == "detail":
            report["status"] = "detail_complete"
        else:
            route = orchestrator.ops.route_finalize(
                tag="planb",
                out_odb=config.save_dir / "planb_routed.odb",
            )
            report["stages"]["route"] = route
            report["metrics"] = route.get("metrics", {})
            report["status"] = (
                "route_complete" if route.get("stage") == "routed" else "route_failed"
            )
        write_report(args.report, report)
        if report["status"].endswith("failed"):
            raise SystemExit(1)
    finally:
        if orchestrator.ctx is not None:
            orchestrator.ctx.session.close()


if __name__ == "__main__":
    main()
