#!/usr/bin/env python3
"""Run the frozen U1 dynamic-graph quick and confirmation screens."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from floorflow.chip_flow.training_runtime import write_json_atomic
from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_u1_dynamic_graph_screen_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/u1_dynamic_graph_screen_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_u1_dynamic_graph_screen_v1.json",
    )
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("quick", "confirmation"),
        default=("quick", "confirmation"),
    )
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_resources(pids: list[int], poll_seconds: int) -> None:
    while any(alive(pid) for pid in pids):
        time.sleep(max(1, poll_seconds))


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed: {log_path}")


def train(
    config: dict[str, Any],
    output_root: Path,
    steps: int,
    arm: str,
    candidate: bool,
    reuse: bool,
) -> Path:
    checkpoint = output_root / arm / "checkpoints" / "final.pt"
    report = output_root / f"{arm}.json"
    if (
        reuse
        and checkpoint.is_file()
        and report.is_file()
        and load(report).get("status") == "complete"
    ):
        return checkpoint
    arch = config["architecture"]
    training = config["training"]
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_chip_slot.py",
        "--cache",
        config["cache"],
        "--train-indices",
        config["train_indices"],
        "--architecture",
        "typed_hierarchy",
        "--arm",
        arm,
        "--out-dir",
        str(output_root),
        "--steps",
        str(steps),
        "--batch-size",
        str(training["batch_size"]),
        "--grad-accum-steps",
        str(training["grad_accum_steps"]),
        "--sampling-mode",
        training["sampling_mode"],
        "--eval-steps",
        str(steps),
        "--curve-per-count",
        str(training["curve_per_count"]),
        "--sampling-steps",
        str(training["sampling_steps"]),
        "--hidden",
        str(arch["hidden"]),
        "--heads",
        str(arch["heads"]),
        "--hierarchy-depth",
        str(arch["depth"]),
        "--typed-layers",
        str(arch["typed_layers"]),
        "--head-dim",
        str(arch["head_dim"]),
        "--projector",
        "--projector-iterations",
        str(arch["projector_iterations"]),
        "--overlap-weight-scale",
        str(training["overlap_weight_scale"]),
        "--hierarchy-bbox-weight",
        str(training["hierarchy_bbox_weight"]),
        "--hierarchy-bbox-decay-steps",
        str(training["hierarchy_bbox_decay_steps"]),
        "--eval-state",
        "ema",
        "--seed",
        str(config["seed"]),
        "--device",
        config["device"],
        "--checkpoint-every",
        str(training["checkpoint_every"]),
    ]
    command.extend(str(value) for value in config.get("training_args", []))
    role = "candidate" if candidate else "control"
    command.extend(
        str(value)
        for value in config.get(f"{role}_extra_training_args", [])
    )
    if config.get("response_sidecar"):
        command.extend(["--response-sidecar", str(config["response_sidecar"])])
    if candidate:
        candidate_flag = config.get("candidate_flag")
        if candidate_flag:
            command.append(str(candidate_flag))
        command.extend(str(value) for value in config.get("candidate_args", []))
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        command.extend(
            [
                "--resume",
                str(ROOT / initial_checkpoint),
                "--resume-weights-only",
            ]
        )
        if config.get("allow_new_params"):
            command.append("--resume-allow-new-params")
    run_checked(command, output_root / f"{arm}.log")
    return checkpoint


def evaluate(
    config: dict[str, Any],
    checkpoint: Path,
    output_root: Path,
    arm: str,
    refiner_steps: int,
    reuse: bool,
) -> Path:
    evaluation = config["evaluation"]
    output = output_root / f"{arm}_refiner{refiner_steps}.json"
    if (
        reuse
        and output.is_file()
        and output.with_name(f"{output.stem}_cases.json").is_file()
        and load(output).get("status") == "single_setting_diagnostic"
    ):
        return output
    command = [
        "uv",
        "run",
        "python",
        "scripts/evaluate_chip_refiner.py",
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(ROOT / config["cache"] / "dev"),
        "--case-indices",
        str(ROOT / config["case_indices"]),
        "--device",
        config["device"],
        "--nfe",
        str(evaluation["nfe"]),
        "--refiner-steps",
        str(refiner_steps),
        "--refiner-lr",
        str(evaluation["refiner_lr"]),
        "--refiner-schedule",
        evaluation["refiner_schedule"],
        "--projector-iterations",
        str(evaluation["projector_iterations"]),
        "--legalizer-time-limit-ms",
        str(evaluation["legalizer_time_limit_ms"]),
        "--seed",
        str(evaluation.get("seed", 19)),
        "--use-ema",
        "--deterministic",
        "--output",
        str(output),
    ]
    for name, value in evaluation["refiner_weights"].items():
        command.extend(["--refiner-weight", f"{name}={value}"])
    command.extend(
        [
            "--post-refiner-projector-iterations",
            str(evaluation.get("post_refiner_projector_iterations", 0)),
            "--post-refiner-projector-strength",
            str(evaluation.get("post_refiner_projector_strength", 1.0)),
        ]
    )
    if "projector_boundary_axes" in evaluation:
        command.append(
            "--projector-boundary-axes"
            if evaluation["projector_boundary_axes"]
            else "--no-projector-boundary-axes"
        )
    run_checked(command, output.with_suffix(".log"))
    return output


def compare(
    config: dict[str, Any],
    control: Path,
    candidate: Path,
    refiner_steps: int,
) -> dict[str, Any]:
    case_key = f"nfe4_refine{refiner_steps}"
    control_rows = load(control.with_name(f"{control.stem}_cases.json"))[
        case_key
    ]
    candidate_rows = load(candidate.with_name(f"{candidate.stem}_cases.json"))[
        case_key
    ]
    metric_names = (
        "legalized_hard_cost",
        "legalized_v_rel",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_area_gap",
        "legalized_grouping",
        "legalized_boundary",
        "legalized_rho_mae",
        "legalizer_rung",
        "legalizer_solves",
        "legalizer_ms",
    )
    metrics = {
        name: paired_delta(
            candidate_rows,
            control_rows,
            name,
            iterations=10000,
            seed=20261100 + index,
            lower_is_better=True,
        )
        for index, name in enumerate(metric_names)
    }
    relative = {
        name: metrics[name]["relative_delta"]
        for name in metrics
    }
    gate = config["promotion"]
    primary = {
        name: relative[name]
        for name in ("legalized_hard_cost", "legalized_v_rel")
    }
    no_regression = all(
        value <= float(gate["max_relative_regression"])
        for value in primary.values()
    )
    improvement = min(primary.values()) <= -float(
        gate["min_relative_improvement_in_at_least_one_metric"]
    )
    return {
        "metrics": metrics,
        "relative": relative,
        "primary": primary,
        "advance": no_regression and improvement,
    }


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    config = load(args.config)
    state = {
        "version": config["version"],
        "status": "waiting_for_resources" if args.wait_pid else "running",
        "candidate": config["candidate"],
        "wait_pids": args.wait_pid,
        "stages": [],
    }
    write_json_atomic(args.report, state)
    wait_for_resources(args.wait_pid, args.poll_seconds)
    state["status"] = "running"
    write_json_atomic(args.report, state)
    for stage_name in args.stages:
        stage = config[stage_name]
        refiner_steps = int(stage["refiner_steps"])
        controls = {}
        candidates = {}
        for arm, is_candidate in (("control", False), ("candidate", True)):
            arm_name = f"{arm}_{stage_name}"
            checkpoint = train(
                config,
                args.output_root,
                int(stage["steps"]),
                arm_name,
                is_candidate,
                args.reuse,
            )
            evaluation = evaluate(
                config,
                checkpoint,
                args.output_root,
                arm_name,
                refiner_steps,
                args.reuse,
            )
            (controls if not is_candidate else candidates)[arm] = {
                "checkpoint": str(checkpoint),
                "evaluation": str(evaluation),
            }
        decision = compare(
            config,
            ROOT / controls["control"]["evaluation"],
            ROOT / candidates["candidate"]["evaluation"],
            refiner_steps,
        )
        state["stages"].append(
            {
                "stage": stage_name,
                "control": controls["control"],
                "candidate": candidates["candidate"],
                "decision": decision,
            }
        )
        write_json_atomic(args.report, state)
        if not decision["advance"]:
            state["status"] = f"stopped_at_failed_{stage_name}_gate"
            write_json_atomic(args.report, state)
            return
    state["status"] = "complete_confirmation_pass"
    write_json_atomic(args.report, state)
    print(json.dumps({"status": state["status"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
