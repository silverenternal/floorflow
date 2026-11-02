#!/usr/bin/env python3
"""Run one refiner1600 diagnostic for the final completed training-ladder arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ladder-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
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


def last_arm(report: dict) -> dict:
    rows = report.get("steps", [])
    if not rows:
        raise ValueError("ladder report has no completed arms")
    return rows[-1]


def main() -> None:
    args = parse_args()
    state = {
        "version": "chip-sota-refiner1600-followup-v1",
        "status": "waiting_for_resources",
        "supervisor_pid": os.getpid(),
        "wait_pids": args.wait_pid,
        "config": str(args.config),
        "ladder_report": str(args.ladder_report),
        "output_root": str(args.output_root),
    }
    write_json_atomic(args.output, state)
    wait_for_pids(args.wait_pid, args.poll_seconds)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    ladder = json.loads(args.ladder_report.read_text(encoding="utf-8"))
    selected = last_arm(ladder)
    steps = int(selected["steps"])
    checkpoint = args.output_root / f"d192_steps{steps}" / "checkpoints" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    eval_config = json.loads(json.dumps(config))
    eval_config["evaluation"]["refiner_steps"] = 1600
    evaluation = args.output_root / f"eval_{steps}_refine1600.json"
    state.update(
        {
            "status": "running",
            "steps": steps,
            "checkpoint": str(checkpoint),
            "evaluation": str(evaluation),
        }
    )
    write_json_atomic(args.output, state)

    from scripts.run_chip_sota_training_ladder import (
        evaluation_command,
        run_checked,
    )

    command = evaluation_command(eval_config, checkpoint, evaluation)
    run_checked(command, args.output_root / f"eval_{steps}_refine1600.log")

    analysis = args.output_root / f"eval_{steps}_refine1600_highblock.json"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/analyze_chip_sota_highblock_ladder.py",
            "--ladder-report",
            str(args.ladder_report),
            "--output-root",
            str(args.output_root),
            "--result-key",
            "nfe4_refine1600",
            "--steps",
            str(steps),
            "--evaluation-template",
            "eval_{steps}_refine1600.json",
            "--output",
            str(analysis),
        ],
        cwd=ROOT,
        check=True,
    )
    state.update(
        {
            "status": "complete",
            "analysis": str(analysis),
        }
    )
    write_json_atomic(args.output, state)
    print(json.dumps(state))


if __name__ == "__main__":
    main()
