#!/usr/bin/env python3
"""Run a frozen ChiPBench DEF-to-model-to-DEF adapter screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_chipbench_holdout_manifest_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", default="candidate")
    parser.add_argument("--cases", nargs="+", default=["VeriGPU", "or1200", "dft68"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--refiner-steps", type=int, default=0)
    parser.add_argument("--refiner-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--refiner-schedule",
        choices=("linear", "cosine", "constant"),
        default="linear",
    )
    parser.add_argument("--refiner-weight", action="append", default=[])
    parser.add_argument(
        "--refiner-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--refiner-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--refiner-chunk", type=int, default=5)
    parser.add_argument("--projector-iterations", type=int, default=8)
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--portfolio-projector-iterations", type=int, default=32)
    parser.add_argument(
        "--legalizer-relation",
        choices=("restored", "expanded"),
        default="restored",
    )
    parser.add_argument("--macro-halo-um", type=float, default=0.0)
    parser.add_argument("--coordinate-grid-dbu", type=int, default=1)
    parser.add_argument("--reference-pin-alignment", action="store_true")
    parser.add_argument("--pin-grid-dbu", type=int, default=280)
    parser.add_argument("--force-shelf", action="store_true")
    parser.add_argument("--position-aware-shelf", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--protocol-tag", default="chipbench_screen_v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chipbench_screen_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chipbench_external_screen_v1.json",
    )
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for(pids: list[int], poll_seconds: int) -> None:
    while any(alive(pid) for pid in pids):
        time.sleep(max(1, poll_seconds))


def placement_command(
    args: argparse.Namespace,
    row: dict[str, Any],
    report_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_chip_external_placement.py"),
        "--case",
        row["id"],
        "--checkpoint",
        str(args.checkpoint),
        "--arm",
            getattr(args, "arm", "candidate"),
        "--seed",
        str(args.seed),
        "--def",
        str(ROOT / row["pre_place_def"]),
        "--device",
        args.device,
        "--nfe",
        str(args.nfe),
        "--refiner-steps",
        str(args.refiner_steps),
        "--refiner-lr",
            str(getattr(args, "refiner_lr", 1.0e-3)),
        "--refiner-schedule",
            getattr(args, "refiner_schedule", "linear"),
        "--refiner-optimizer",
            getattr(args, "refiner_optimizer", "adam"),
        "--refiner-chunk",
            str(getattr(args, "refiner_chunk", 5)),
        "--projector-iterations",
        str(args.projector_iterations),
        "--legalizer-relation",
            getattr(args, "legalizer_relation", "restored"),
        "--macro-halo-um",
            str(getattr(args, "macro_halo_um", 0.0)),
        "--instance-hierarchy",
        "--preserve-reference-orientation",
        "--protocol-tag",
        args.protocol_tag,
        "--output",
        str(report_path),
    ]
    if getattr(args, "refiner_shape", False):
        command.append("--refiner-shape")
    if getattr(args, "portfolio", False):
        command.extend(
            [
                "--portfolio",
                "--portfolio-projector-iterations",
                str(args.portfolio_projector_iterations),
            ]
        )
    if getattr(args, "deterministic", False):
        command.append("--deterministic")
    if getattr(args, "force_shelf", False):
        command.append("--force-shelf")
    if getattr(args, "position_aware_shelf", False):
        command.append("--position-aware-shelf")
    for weight in getattr(args, "refiner_weight", []):
        command.extend(["--refiner-weight", weight])
    for path in row["macro_lefs"]:
        command.extend(["--macro-lef", str(ROOT / path)])
    return command


def materialize_command(
    args: argparse.Namespace,
    row: dict[str, Any],
    boxes: Path,
    output_def: Path,
    report_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/materialize_chipbench_def.py"),
        "--source-def",
        str(ROOT / row["pre_place_def"]),
        "--candidate-boxes",
        str(boxes),
        "--output-def",
        str(output_def),
        "--remove-blockages",
        "--coordinate-grid-dbu",
        str(getattr(args, "coordinate_grid_dbu", 1)),
        "--report",
        str(report_path),
    ]
    for path in row["macro_lefs"]:
        command.extend(["--verify-macro-lef", str(ROOT / path)])
    if getattr(args, "reference_pin_alignment", False):
        command.extend(
            [
                "--reference-def",
                str(ROOT / row["reference_macro_def"]),
                "--pin-grid-dbu",
                str(args.pin_grid_dbu),
                "--pin-rollback-conflicts",
            ]
        )
    return command


def run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    if args.nfe <= 0 or args.refiner_steps < 0:
        raise ValueError("nfe must be positive and refiner steps non-negative")
    manifest = load(args.manifest)
    by_id = {
        row["id"]: row
        for row in manifest["designs"]
        if row.get("pre_place_def")
    }
    missing = sorted(set(args.cases) - set(by_id))
    if missing:
        raise ValueError(f"cases are absent from frozen ChiPBench manifest: {missing}")
    queue = {
        "version": "chipbench-external-screen-v1",
        "status": "waiting_for_resources" if args.wait_pid else "running",
        "manifest": str(args.manifest.relative_to(ROOT)),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "nfe": args.nfe,
        "refiner_steps": args.refiner_steps,
        "refiner_lr": args.refiner_lr,
        "refiner_schedule": args.refiner_schedule,
        "refiner_weights": args.refiner_weight,
        "refiner_shape": args.refiner_shape,
        "refiner_optimizer": args.refiner_optimizer,
        "legalizer_relation": args.legalizer_relation,
        "portfolio": bool(args.portfolio),
        "portfolio_projector_iterations": int(
            args.portfolio_projector_iterations
        ),
        "macro_halo_um": args.macro_halo_um,
        "coordinate_grid_dbu": args.coordinate_grid_dbu,
        "reference_pin_alignment": bool(args.reference_pin_alignment),
        "pin_grid_dbu": int(args.pin_grid_dbu),
        "seed": args.seed,
        "cases": args.cases,
        "runs": [],
        "policy": "Adapter plumbing only; no quality or routed-PPA claim.",
    }
    write(args.report, queue)
    wait_for(args.wait_pid, args.poll_seconds)
    queue["status"] = "running"
    write(args.report, queue)
    for case in args.cases:
        row = by_id[case]
        case_root = args.output_root / case
        placement_report = case_root / "placement.json"
        placement_log = case_root / "placement.log"
        placement_code = run(
            placement_command(args, row, placement_report),
            placement_log,
        )
        placement = load(placement_report) if placement_report.is_file() else {}
        boxes = placement.get("outputs", {}).get("boxes")
        materialize_report = case_root / "materialize.json"
        output_def = case_root / f"{args.arm}_seed{args.seed}.def"
        materialize_code = None
        if boxes:
            materialize_code = run(
                materialize_command(
                    args,
                    row,
                    ROOT / boxes,
                    output_def,
                    materialize_report,
                ),
                case_root / "materialize.log",
            )
        materialized = (
            load(materialize_report)
            if materialize_report.is_file()
            else {}
        )
        roundtrip = materialized.get("roundtrip") or {}
        checks = {
            "placement_pass": placement.get("status") == "pass",
            "materialize_pass": roundtrip.get("status") == "pass",
            "same_macro_count": (
                placement.get("graph", {}).get("macros")
                == roundtrip.get("expected_macros")
                == roundtrip.get("materialized_macros")
                == row["macro_instance_count"]
            ),
        }
        queue["runs"].append(
            {
                "case": case,
                "placement_exit_code": placement_code,
                "materialize_exit_code": materialize_code,
                "placement_report": str(placement_report.relative_to(ROOT)),
                "materialize_report": str(materialize_report.relative_to(ROOT)),
                "output_def": (
                    str(output_def.relative_to(ROOT))
                    if output_def.is_file()
                    else None
                ),
                "roundtrip": roundtrip,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
        write(args.report, queue)
    queue["status"] = (
        "complete"
        if all(row["status"] == "pass" for row in queue["runs"])
        else "failed"
    )
    write(args.report, queue)
    print(json.dumps({"status": queue["status"], "report": str(args.report)}))
    if queue["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
