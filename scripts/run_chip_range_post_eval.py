#!/usr/bin/env python3
"""Run paired and block-stratified analysis after the range control finishes."""

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
from scripts.run_chip_v46_response_queue import run_logged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-pid", type=int, default=1831103)
    parser.add_argument("--worker-pid", type=int, default=1831141)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument(
        "--range-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_training_range_control_v1.json",
    )
    parser.add_argument(
        "--candidate-eval",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/sota_range_control_v1/eval_20000.json",
    )
    parser.add_argument(
        "--control-eval",
        type=Path,
        default=(
            ROOT
            / "outputs/chip_hierarchy/sota_subset_control_v2/eval_20000_cuda1.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/sota_range_control_v1",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "outputs/reports/chip_range_post_eval_queue_v1.json",
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


def wait_for_processes(pids: list[int], poll_seconds: int) -> float:
    started = time.perf_counter()
    while any(process_running(pid) for pid in pids):
        time.sleep(max(1, poll_seconds))
    return time.perf_counter() - started


def comparison_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_chip_training_frontier_comparison.py"),
        "--baseline",
        str(args.control_eval),
        "--candidate",
        str(args.candidate_eval),
        "--output",
        str(ROOT / "outputs/reports/chip_range_control_comparison_v1.json"),
        "--scope",
        "same-device dev32: 21-120 frozen subset vs clean 21-60 frozen subset",
    ]


def highblock_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/analyze_chip_sota_highblock_ladder.py"),
        "--ladder-report",
        str(args.range_report),
        "--output-root",
        str(args.output_root),
        "--steps",
        "20000",
        "--output",
        str(ROOT / "outputs/reports/chip_range20k_highblock_analysis_v1.json"),
    ]


def main() -> None:
    args = parse_args()
    state = {
        "version": "chip-range-post-eval-queue-v1",
        "status": "waiting_for_range_control",
        "wait_pids": [args.supervisor_pid, args.worker_pid],
        "runs": [],
    }
    write_json_atomic(args.state, state)
    state["wait_seconds"] = wait_for_processes(
        [args.supervisor_pid, args.worker_pid],
        args.poll_seconds,
    )
    missing = [
        str(path)
        for path in (args.range_report, args.candidate_eval, args.control_eval)
        if not path.is_file()
    ]
    if missing:
        state["status"] = "blocked_missing_artifacts"
        state["missing"] = missing
        write_json_atomic(args.state, state)
        raise SystemExit(1)
    report = json.loads(args.range_report.read_text(encoding="utf-8"))
    state["range_status"] = report.get("status")
    state["status"] = "running"
    write_json_atomic(args.state, state)
    for name, command in (
        ("paired_comparison", comparison_command(args)),
        ("highblock_strata", highblock_command(args)),
    ):
        result = run_logged(
            command,
            args.state.with_name(f"{args.state.stem}_{name}.log"),
        )
        state["runs"].append({"name": name, **result})
        write_json_atomic(args.state, state)
    state["status"] = (
        "complete"
        if all(run["exit_code"] == 0 for run in state["runs"])
        else "completed_with_failures"
    )
    write_json_atomic(args.state, state)
    print(json.dumps(state))
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
