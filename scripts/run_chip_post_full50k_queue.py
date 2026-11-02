#!/usr/bin/env python3
"""Serialize M10 and M6 follow-ups behind the active full-50k supervisor."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.run_chip_v46_response_queue import run_logged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, action="append")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/sota_full50k_exposure_v1"
            / "checkpoints/final.pt"
        ),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_sota_training_full50k_exposure_v1.json"
        ),
    )
    parser.add_argument(
        "--fallback-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/v7_bbox_anneal_20k"
            / "typed_hierarchy_bbox_anneal_seed19_20k/checkpoints/final.pt"
        ),
    )
    parser.add_argument(
        "--m10-output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_m10_projection_consistent_flow_dev32_v1.json"
        ),
    )
    parser.add_argument(
        "--m6-output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_response_repair_queue.json",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "outputs/reports/chip_post_full50k_queue_v1.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-m6", action="store_true")
    parser.add_argument("--skip-m10", action="store_true")
    return parser.parse_args()


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_processes(pids: list[int], poll_seconds: int) -> float:
    started = time.perf_counter()
    while any(process_running(pid) for pid in pids):
        time.sleep(max(1, poll_seconds))
    return time.perf_counter() - started


def m10_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_chip_m10_projection_flow_screen.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--stage",
        "quick",
        "--device",
        "cuda:0",
        "--output",
        str(args.m10_output),
    ]


def m6_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_chip_v46_response_repair.py"),
        "--output",
        str(args.m6_output),
    ]


def update_state(path: Path, state: dict[str, Any]) -> None:
    write_json_atomic(path, state)


def select_m10_checkpoint(
    training_report: Path,
    trained_checkpoint: Path,
    fallback_checkpoint: Path,
) -> tuple[Path, str]:
    if not training_report.is_file():
        raise ValueError(f"missing full-50k training report: {training_report}")
    report = json.loads(training_report.read_text(encoding="utf-8"))
    status = report.get("status")
    if status == "complete":
        if not trained_checkpoint.is_file():
            raise ValueError(f"missing trained checkpoint: {trained_checkpoint}")
        return trained_checkpoint, "full50k_passed_advance_gate"
    if status == "stopped_at_failed_advance_gate":
        if not fallback_checkpoint.is_file():
            raise ValueError(f"missing fallback checkpoint: {fallback_checkpoint}")
        return fallback_checkpoint, "full50k_failed_advance_gate_use_frozen_reference"
    raise ValueError(f"unresolved full-50k status: {status}")


def main() -> None:
    args = parse_args()
    wait_pids = args.wait_pid or [1719776, 1719812]
    started = time.perf_counter()
    state: dict[str, Any] = {
        "version": "chip-post-full50k-queue-v1",
        "status": "waiting_for_full50k",
        "wait_pids": wait_pids,
        "checkpoint": str(args.checkpoint),
        "fallback_checkpoint": str(args.fallback_checkpoint),
        "training_report": str(args.training_report),
        "skip_m10": bool(args.skip_m10),
        "skip_m6": bool(args.skip_m6),
        "runs": [],
    }
    update_state(args.state, state)
    state["wait_seconds"] = wait_for_processes(
        wait_pids,
        args.poll_seconds,
    )
    if args.skip_m10:
        selected_checkpoint = None
        selection_reason = "skipped_rejected_m10"
    else:
        try:
            selected_checkpoint, selection_reason = select_m10_checkpoint(
                args.training_report,
                args.checkpoint,
                args.fallback_checkpoint,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            state["status"] = "m10_blocked_checkpoint_selection"
            state["m10_selection_error"] = str(error)
            selected_checkpoint = None
            selection_reason = "blocked"
    state["m10_checkpoint"] = (
        str(selected_checkpoint) if selected_checkpoint is not None else None
    )
    state["m10_selection_reason"] = selection_reason
    state["status"] = "running"
    update_state(args.state, state)
    commands = []
    if not args.skip_m10 and selected_checkpoint is not None:
        args.checkpoint = selected_checkpoint
        commands.append(("m10_projection_flow_quick", m10_command(args)))
    if not args.skip_m6:
        commands.append(("m6_response_repair", m6_command(args)))
    if args.dry_run:
        state["status"] = "dry_run"
        state["planned_commands"] = commands
        state["elapsed_seconds"] = time.perf_counter() - started
        update_state(args.state, state)
        print(json.dumps(state))
        return
    for name, command in commands:
        result = run_logged(
            command,
            args.state.with_name(f"{args.state.stem}_{name}.log"),
        )
        state["runs"].append({"name": name, **result})
        update_state(args.state, state)
    m10_blocked = not args.skip_m10 and selected_checkpoint is None
    state["status"] = (
        "complete"
        if not m10_blocked
        and all(run["exit_code"] == 0 for run in state["runs"])
        else "completed_with_failures"
    )
    state["elapsed_seconds"] = time.perf_counter() - started
    update_state(args.state, state)
    print(json.dumps(state))
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
