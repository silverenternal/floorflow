#!/usr/bin/env python3
"""Run routed-PPA sensitivity pairs after prerequisite jobs finish."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--placements",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_collapse_sensitivity.json",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--num-cores", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_collapse_ppa.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_alive(pid: int) -> bool:
    try:
        return bool(Path(f"/proc/{pid}/cmdline").read_bytes())
    except FileNotFoundError:
        return False


def wait_for(processes: list[int], poll_seconds: int) -> None:
    while any(process_alive(pid) for pid in processes):
        time.sleep(max(1, poll_seconds))


def flow_command(
    args: argparse.Namespace,
    case: str,
    arm: str,
    tcl: Path,
) -> list[str]:
    slug = case.replace("/", "__")
    variant = f"ppa_v45_collapse_sensitivity_{arm}_s19"
    output = (
        ROOT
        / "outputs/reports/v45_collapse_ppa"
        / slug
        / f"{arm}_seed19.json"
    )
    return [
        sys.executable,
        str(ROOT / "scripts/run_chip_ppa_external.py"),
        "--case",
        case,
        "--manifest",
        str(ROOT / "outputs/reports/chip_external_inventory_v45_expanded_audit.json"),
        "--variant",
        variant,
        "--macro-placement-tcl",
        str(tcl),
        "--run-flow",
        "--flow-target",
        "finish",
        "--num-cores",
        str(args.num_cores),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--smoke-output",
        str(output),
    ]


def run_arm(args: argparse.Namespace, case: str, arm: str, tcl: Path) -> dict[str, Any]:
    command = flow_command(args, case, arm, tcl)
    output = Path(command[command.index("--smoke-output") + 1])
    log = output.with_suffix(".log")
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = load(output) if output.is_file() else {}
    return {
        "case": case,
        "arm": arm,
        "exit_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "report": str(output.relative_to(ROOT)),
        "runner_log": str(log.relative_to(ROOT)),
        "status": report.get("status"),
        "metrics": report.get("metric_audit", {}).get("values", {}),
    }


def aggregate_pair(case: str, candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("status") != "pass" or control.get("status") != "pass":
        return {"case": case, "status": "failed"}
    cv = candidate["metrics"]
    ctl = control["metrics"]
    candidate_wl = cv.get("detailedroute__route__wirelength")
    control_wl = ctl.get("detailedroute__route__wirelength")
    return {
        "case": case,
        "status": "completed",
        "routed_wirelength": {
            "candidate": candidate_wl,
            "control": control_wl,
            "relative": candidate_wl / control_wl - 1.0,
        },
        "congestion": {
            "candidate": cv.get("globalplace__gpl__routability__congestion"),
            "control": ctl.get("globalplace__gpl__routability__congestion"),
        },
        "setup_ws": {
            "candidate": cv.get("finish__timing__setup__ws"),
            "control": ctl.get("finish__timing__setup__ws"),
        },
        "drc": {
            "candidate": cv.get("detailedroute__route__drc_errors"),
            "control": ctl.get("detailedroute__route__drc_errors"),
        },
        "power_relative": (
            cv.get("finish__power__total") / ctl.get("finish__power__total") - 1.0
            if cv.get("finish__power__total") and ctl.get("finish__power__total")
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    placements = load(args.placements)
    by_case = {row["case"]: row for row in placements["cases"]}
    selected = args.case or ["sky130hd/microwatt"]
    queue = {
        "version": "chip-v45-collapse-ppa-sensitivity-v1",
        "status": "waiting" if args.wait_pid else "running",
        "wait_pids": args.wait_pid,
        "cases": selected,
        "runs": [],
        "comparisons": [],
        "policy": (
            "Corrected routed PPA is a sensitivity cohort. It does not replace "
            "the original observed v45 outcomes."
        ),
    }
    write(args.output, queue)
    wait_for(args.wait_pid, args.poll_seconds)
    queue["status"] = "running"
    write(args.output, queue)
    for case in selected:
        row = by_case[case]
        if not row["distinct_after_position_aware_shelf"]:
            raise ValueError(f"selected pair is still collapsed: {case}")
        arm_rows = {}
        for arm in ("candidate", "control"):
            tcl = ROOT / row["arms"][arm]["tcl"]
            result = run_arm(args, case, arm, tcl)
            arm_rows[arm] = result
            queue["runs"].append(result)
            write(args.output, queue)
        queue["comparisons"].append(
            aggregate_pair(case, arm_rows["candidate"], arm_rows["control"])
        )
        write(args.output, queue)
    queue["status"] = (
        "complete"
        if all(row.get("status") == "completed" for row in queue["comparisons"])
        else "failed"
    )
    write(args.output, queue)
    print(json.dumps({"status": queue["status"], "output": str(args.output)}))
    if queue["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
