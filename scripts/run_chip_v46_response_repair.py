#!/usr/bin/env python3
"""Repair incomplete M6 response arms after the serialized training chain."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.run_chip_v46_response_queue import arm_command, run_logged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_response_repair_queue.json",
    )
    return parser.parse_args()


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pids(pids: list[int], poll_seconds: int) -> None:
    while any(process_running(pid) for pid in pids):
        time.sleep(max(1, poll_seconds))


def arm_report_path(arm: str, proposal_count: int) -> Path:
    return (
        ROOT
        / "outputs/reports"
        / f"chip_v46_analytical_search_{arm}_n{proposal_count}_positioned_all.json"
    )


def arm_complete(arm: str, proposal_count: int) -> bool:
    path = arm_report_path(arm, proposal_count)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    state = {
        "version": "chip-v46-response-repair-queue-v1",
        "status": "waiting_for_training_chain",
        "wait_pids": args.wait_pid,
        "runs": [],
    }
    write_json_atomic(args.output, state)
    wait_for_pids(args.wait_pid, args.poll_seconds)
    state["status"] = "running"
    write_json_atomic(args.output, state)
    targets = (
        ("candidate", 1),
        ("candidate", 2),
        ("control", 1),
        ("control", 2),
    )
    for arm, proposal_count in targets:
        if arm_complete(arm, proposal_count):
            state["runs"].append(
                {
                    "arm": arm,
                    "proposal_count": proposal_count,
                    "status": "skipped_already_complete",
                }
            )
            continue
        result = run_logged(
            arm_command(arm, proposal_count),
            ROOT
            / "outputs/reports"
            / f"chip_v46_response_repair_{arm}_n{proposal_count}.log",
        )
        state["runs"].append(
            {
                "arm": arm,
                "proposal_count": proposal_count,
                "status": (
                    "complete"
                    if result["exit_code"] == 0
                    and arm_complete(arm, proposal_count)
                    else "failed"
                ),
                **result,
            }
        )
        write_json_atomic(args.output, state)
    report = run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/build_chip_v46_analytical_search_report.py"),
            "--output",
            str(ROOT / "outputs/reports/chip_v46_analytical_search_report.json"),
        ],
        ROOT / "outputs/reports/chip_v46_analytical_search_repair_report.log",
    )
    state["report_build"] = report
    state["status"] = (
        "complete"
        if report["exit_code"] == 0
        and all(row["status"] in ("complete", "skipped_already_complete") for row in state["runs"])
        else "failed"
    )
    state["elapsed_seconds"] = time.perf_counter() - started
    write_json_atomic(args.output, state)
    print(json.dumps(state))
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
