#!/usr/bin/env python3
"""Wait for the active ORFS run, then execute the frozen response-only queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_response_queue.json",
    )
    return parser.parse_args()


def active_orfs_process(pid: int) -> bool:
    command = Path(f"/proc/{pid}/cmdline")
    try:
        text = command.read_bytes().decode(errors="replace")
    except FileNotFoundError:
        return False
    return "openroad" in text and "detail_route.tcl" in text


def arm_command(arm: str, proposal_count: int) -> list[str]:
    checkpoint = (
        ROOT
        / (
            "outputs/chip_hierarchy/v7_bbox_anneal_20k/"
            "typed_hierarchy_bbox_anneal_seed19_20k/checkpoints/final.pt"
        )
        if arm == "candidate"
        else ROOT
        / (
            "outputs/chip_trinity/rescue_20k/"
            "trinity_token_seed19_20k/checkpoints/final.pt"
        )
    )
    return [
        sys.executable,
        str(ROOT / "scripts/evaluate_chip_analytical_search.py"),
        "--arm",
        arm,
        "--checkpoint",
        str(checkpoint),
        "--proposal-count",
        str(proposal_count),
        "--case",
        "nangate45/cva6",
        "--case",
        "nangate45/bp_be_top",
        "--device",
        "cuda:0",
        "--response-only",
        "--output",
        str(
            ROOT
            / "outputs/reports"
            / f"chip_v46_analytical_search_{arm}_n{proposal_count}_positioned_all.json"
        ),
    ]


def run_logged(command: list[str], log_path: Path) -> dict:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    queue = {
        "version": "chip-v46-response-queue-v1",
        "status": "waiting_for_orfs",
        "wait_pid": args.wait_pid,
        "poll_seconds": args.poll_seconds,
        "runs": [],
    }
    write(args.output, queue)
    while active_orfs_process(args.wait_pid):
        time.sleep(max(1, args.poll_seconds))
    queue["status"] = "running_responses"
    queue["wait_seconds"] = time.perf_counter() - started
    write(args.output, queue)
    for arm in ("candidate", "control"):
        for proposal_count in (1, 2):
            result = run_logged(
                arm_command(arm, proposal_count),
                ROOT
                / "outputs/reports"
                / f"chip_v46_response_{arm}_n{proposal_count}.log",
            )
            queue["runs"].append(
                {
                    "arm": arm,
                    "proposal_count": proposal_count,
                    **result,
                }
            )
            write(args.output, queue)
    report = run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/build_chip_v46_analytical_search_report.py"),
            "--output",
            str(ROOT / "outputs/reports/chip_v46_analytical_search_report.json"),
        ],
        ROOT / "outputs/reports/chip_v46_analytical_search_report.log",
    )
    queue["report_build"] = report
    queue["status"] = (
        "complete"
        if all(row["exit_code"] == 0 for row in queue["runs"])
        and report["exit_code"] == 0
        else "failed"
    )
    queue["elapsed_seconds"] = time.perf_counter() - started
    write(args.output, queue)
    print(json.dumps({"status": queue["status"], "output": str(args.output)}))
    if queue["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
