#!/usr/bin/env python3
"""Run the predeclared S3 capacity screen with budget-class gating."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.run_chip_sota_training_ladder import (
    evaluation_command,
    evaluation_metrics,
    process_running,
    run_checked,
    should_advance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs/chip_sota_capacity_screen_v1.json",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/chip_sota_training_compute_ladder_v1.json",
    )
    parser.add_argument("--phase", choices=("quick", "confirmation"), default="quick")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/capacity_screen_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_capacity_screen_v1.json",
    )
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--diagnostic-refiner-steps", type=int, default=1600)
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument(
        "--base-ladder-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_training_compute_ladder_v1.json",
    )
    parser.add_argument(
        "--range-ladder-report",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_training_range_compute_ladder_v1.json",
    )
    return parser.parse_args()


def wait_for_pids(pids: list[int], poll_seconds: int) -> None:
    while any(process_running(pid, []) for pid in pids):
        time.sleep(max(1, poll_seconds))


def training_command(
    base: dict,
    phase: dict,
    arm: dict,
    output_root: Path,
) -> list[str]:
    training = base["training"]
    width = int(arm["width"])
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_chip_slot.py",
        "--cache",
        str(ROOT / phase["cache"]),
        "--train-indices",
        str(ROOT / phase["train_indices"]),
        "--architecture",
        training["architecture"],
        "--arm",
        str(arm["id"]),
        "--out-dir",
        str(output_root),
        "--steps",
        str(arm["steps"]),
        "--batch-size",
        str(training["micro_batch"]),
        "--grad-accum-steps",
        str(training["grad_accum_steps"]),
        "--sampling-mode",
        training["sampling_mode"],
        "--eval-steps",
        str(arm["steps"]),
        "--curve-per-count",
        str(training["curve_per_count"]),
        "--sampling-steps",
        str(training["sampling_steps"]),
        "--hidden",
        str(width),
        "--heads",
        str(max(1, width // 64)),
        "--hierarchy-depth",
        str(training["hierarchy_depth"]),
        "--typed-layers",
        str(training["typed_layers"]),
        "--head-dim",
        str(training["head_dim"]),
        "--projector",
        "--projector-iterations",
        str(training["projector_iterations"]),
        "--overlap-weight-scale",
        str(training["overlap_weight_scale"]),
        "--hierarchy-bbox-weight",
        str(training["hierarchy_bbox_weight"]),
        "--hierarchy-bbox-decay-steps",
        str(training["hierarchy_bbox_decay_steps"]),
        "--eval-state",
        training["eval_state"],
        "--seed",
        str(phase["seed"]),
        "--device",
        phase["device"],
        "--checkpoint-every",
        str(training["checkpoint_every"]),
    ]
    return command


def arm_gate(control: dict, candidate: dict) -> dict:
    gate = should_advance(
        control,
        candidate,
        max_relative_regression=0.002,
        min_relative_improvement=0.005,
    )
    gate["passes_budget_class"] = bool(gate["advance"])
    return gate


def phase_payload(plan: dict, phase: str) -> dict:
    return plan["quick_screen" if phase == "quick" else "confirmation"]


def validate_ladder_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    steps = report.get("steps", [])
    if not steps:
        raise ValueError(f"{path} has no completed training arm")
    return {
        "path": str(path),
        "status": report.get("status"),
        "completed_steps": [int(row["steps"]) for row in steps],
    }


def resize_confirmation_phase(phase: dict, base_steps: int) -> dict:
    resized = json.loads(json.dumps(phase))
    resized["base_steps"] = int(base_steps)
    for arm in resized["arms"]:
        ratio = float(arm["measured_block_ratio"])
        arm["steps"] = (
            max(1, round(int(base_steps) / ratio))
            if arm["budget_class"] == "equal-flops"
            else int(base_steps)
        )
    return resized


def run_arm(
    base: dict,
    phase: dict,
    arm: dict,
    output_root: Path,
    diagnostic_refiner_steps: int,
    skip_diagnostic: bool,
) -> dict:
    arm_root = output_root / str(arm["id"])
    command = training_command(base, phase, arm, output_root)
    run_checked(command, output_root / f"{arm['id']}_train.log")
    checkpoint = output_root / str(arm["id"]) / "checkpoints" / "final.pt"
    evaluation = arm_root / "nfe4_refine400.json"
    eval_config = json.loads(json.dumps(base))
    eval_config["cache"] = phase["cache"]
    run_checked(
        evaluation_command(eval_config, checkpoint, evaluation),
        arm_root / "nfe4_refine400.log",
    )
    result = {
        "arm": arm,
        "metrics": evaluation_metrics(evaluation),
        "evaluation": str(evaluation),
    }
    if not skip_diagnostic:
        diagnostic = arm_root / f"nfe4_refine{diagnostic_refiner_steps}.json"
        diagnostic_config = json.loads(json.dumps(eval_config))
        diagnostic_config["evaluation"]["refiner_steps"] = diagnostic_refiner_steps
        run_checked(
            evaluation_command(diagnostic_config, checkpoint, diagnostic),
            arm_root / f"nfe4_refine{diagnostic_refiner_steps}.log",
        )
        result["diagnostic"] = {
            "refiner_steps": diagnostic_refiner_steps,
            "metrics": evaluation_metrics_for_key(
                diagnostic,
                f"nfe4_refine{diagnostic_refiner_steps}",
            ),
            "evaluation": str(diagnostic),
        }
    return result


def evaluation_metrics_for_key(path: Path, key: str) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report["results"][key]
    feasible = float(result["legalized_feasible_fraction"])
    if feasible != 1.0:
        raise ValueError(f"evaluation feasibility is {feasible}, not 1.0")
    return {
        "cost": float(result["legalized_hard_cost"]),
        "v_rel": float(result["legalized_v_rel"]),
        "feasible_fraction": feasible,
    }


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    phase = phase_payload(plan, args.phase)
    state = {
        "version": plan["version"],
        "status": "waiting_for_resources",
        "phase": args.phase,
        "plan": str(args.plan),
        "wait_pids": args.wait_pid,
        "arms": [],
    }
    write_json_atomic(args.report, state)
    wait_for_pids(args.wait_pid, args.poll_seconds)
    state["s1_validation"] = [
        validate_ladder_report(args.base_ladder_report),
        validate_ladder_report(args.range_ladder_report),
    ]
    if args.phase == "confirmation":
        selected_horizon = state["s1_validation"][1]["completed_steps"][-1]
        phase = resize_confirmation_phase(phase, selected_horizon)
        state["selected_s1_horizon"] = selected_horizon
        state["resized_confirmation_steps"] = {
            row["id"]: row["steps"] for row in phase["arms"]
        }
    state["status"] = "running"
    write_json_atomic(args.report, state)

    arm_rows = phase["arms"]
    control_row = next(row for row in arm_rows if row["id"] == "d192-control")
    control = run_arm(
        base,
        phase,
        control_row,
        args.output_root,
        args.diagnostic_refiner_steps,
        args.skip_diagnostic,
    )
    state["arms"].append(control)
    write_json_atomic(args.report, state)
    controls = {"d192-control": control["metrics"]}

    gates = {}
    for row in arm_rows:
        if row["id"] == "d192-control":
            continue
        if row["width"] == 384:
            break
        result = run_arm(
            base,
            phase,
            row,
            args.output_root,
            args.diagnostic_refiner_steps,
            args.skip_diagnostic,
        )
        result["gate"] = arm_gate(controls["d192-control"], result["metrics"])
        gates[row["budget_class"]] = result["gate"]
        state["arms"].append(result)
        write_json_atomic(args.report, state)

    d384_rows = [
        row for row in arm_rows if row["width"] == 384
    ]
    for row in d384_rows:
        if not gates.get(row["budget_class"], {}).get("passes_budget_class", False):
            continue
        result = run_arm(
            base,
            phase,
            row,
            args.output_root,
            args.diagnostic_refiner_steps,
            args.skip_diagnostic,
        )
        result["gate"] = arm_gate(controls["d192-control"], result["metrics"])
        state["arms"].append(result)
        write_json_atomic(args.report, state)
    state["status"] = "complete"
    state["budget_gates"] = gates
    write_json_atomic(args.report, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
