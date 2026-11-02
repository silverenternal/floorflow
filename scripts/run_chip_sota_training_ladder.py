#!/usr/bin/env python3
"""Run the frozen SOTA training-compute ladder after external jobs release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/sota_budget_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_sota_training_compute_ladder_v1.json",
    )
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--wait-token", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def process_running(pid: int, tokens: list[str]) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if tokens and cmdline_path.is_file():
        command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
        if not all(token in command for token in tokens):
            return False
    return True


def wait_for_resources(
    pids: list[int],
    tokens: list[str],
    poll_seconds: int,
) -> None:
    while any(process_running(pid, tokens) for pid in pids):
        time.sleep(max(1, poll_seconds))


def evaluation_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result = report["results"]["nfe4_refine400"]
    feasible = float(result["legalized_feasible_fraction"])
    if feasible != 1.0:
        raise ValueError(f"evaluation feasibility is {feasible}, not 1.0")
    return {
        "cost": float(result["legalized_hard_cost"]),
        "v_rel": float(result["legalized_v_rel"]),
        "feasible_fraction": feasible,
    }


def should_advance(
    previous: dict[str, float],
    current: dict[str, float],
    *,
    max_relative_regression: float,
    min_relative_improvement: float,
) -> dict:
    cost_relative = current["cost"] / previous["cost"] - 1.0
    v_rel_relative = current["v_rel"] / previous["v_rel"] - 1.0
    no_large_regression = (
        cost_relative <= max_relative_regression
        and v_rel_relative <= max_relative_regression
    )
    meaningful_improvement = min(cost_relative, v_rel_relative) <= (
        -min_relative_improvement
    )
    return {
        "cost_relative": cost_relative,
        "v_rel_relative": v_rel_relative,
        "no_large_regression": no_large_regression,
        "meaningful_improvement": meaningful_improvement,
        "advance": no_large_regression and meaningful_improvement,
    }


def training_command(
    config: dict,
    steps: int,
    output_root: Path,
) -> list[str]:
    training = config["training"]
    arm = f"d192_steps{steps}"
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_chip_slot.py",
        "--cache",
        config["cache"],
        "--architecture",
        training["architecture"],
        "--arm",
        arm,
        "--out-dir",
        str(output_root),
        "--steps",
        str(steps),
        "--batch-size",
        str(training["micro_batch"]),
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
        str(training["hidden"]),
        "--heads",
        str(training["heads"]),
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
        str(training["seed"]),
        "--device",
        config["device"],
        "--checkpoint-every",
        str(training["checkpoint_every"]),
    ]
    if config.get("train_indices"):
        command.extend(
            ["--train-indices", str(config["train_indices"])]
        )
    else:
        command.append("--allow-full-data")
    return command


def evaluation_command(
    config: dict,
    checkpoint: Path,
    output: Path,
) -> list[str]:
    evaluation = config["evaluation"]
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
        str(ROOT / evaluation["case_indices"]),
        "--device",
        evaluation["device"],
        "--nfe",
        str(evaluation["nfe"]),
        "--refiner-steps",
        str(evaluation["refiner_steps"]),
        "--refiner-lr",
        str(evaluation["refiner_lr"]),
        "--refiner-schedule",
        evaluation["refiner_schedule"],
        "--projector-iterations",
        str(evaluation["projector_iterations"]),
        "--legalizer-time-limit-ms",
        str(evaluation["legalizer_time_limit_ms"]),
        "--output",
        str(output),
    ]
    if evaluation["use_ema"]:
        command.append("--use-ema")
    if evaluation["deterministic"]:
        command.append("--deterministic")
    for name, value in evaluation["refiner_weights"].items():
        command.extend(["--refiner-weight", f"{name}={value}"])
    return command


def run_checked(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with {completed.returncode}; see {log_path}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("train_indices") and (
        sha256(ROOT / config["train_indices"])
        != config["train_indices_sha256"]
    ):
        raise ValueError("training subset hash mismatch")
    if (
        sha256(ROOT / config["cache"] / "train" / "metadata.json")
        != config["cache_metadata_sha256"]
    ):
        raise ValueError("training cache metadata hash mismatch")
    output_root = args.output_root.resolve()
    records = []
    previous = None
    reference_path = config.get("reference_eval")
    reference = (
        evaluation_metrics(ROOT / reference_path)
        if reference_path
        else None
    )
    state = {
        "version": config["version"],
        "status": "waiting_for_resources",
        "config": str(args.config),
        "train_indices_sha256": config.get("train_indices_sha256"),
        "cache_metadata_sha256": config["cache_metadata_sha256"],
        "supervisor_pid": os.getpid(),
        "wait_pids": args.wait_pid,
        "reference_eval": reference_path,
        "reference_metrics": reference,
        "steps": [],
    }
    write_json_atomic(args.report, state)
    wait_for_resources(
        args.wait_pid,
        args.wait_token,
        args.poll_seconds,
    )
    state["status"] = "running"
    write_json_atomic(args.report, state)
    for steps in config["steps"]:
        train_log = output_root / f"train_{steps}.log"
        train_command = training_command(config, steps, output_root)
        run_checked(train_command, train_log)
        checkpoint = output_root / f"d192_steps{steps}" / "checkpoints" / "final.pt"
        eval_output = output_root / f"eval_{steps}.json"
        run_checked(
            evaluation_command(config, checkpoint, eval_output),
            output_root / f"eval_{steps}.log",
        )
        metrics = evaluation_metrics(eval_output)
        decision = None
        if previous is not None or reference is not None:
            gate = config["advance_gate"]
            decision = should_advance(
                previous["metrics"] if previous is not None else reference,
                metrics,
                max_relative_regression=float(
                    gate["max_relative_regression"]
                ),
                min_relative_improvement=float(
                    gate["min_relative_improvement_in_at_least_one_metric"]
                ),
            )
            decision["reference"] = (
                "previous_ladder_arm"
                if previous is not None
                else str(reference_path)
            )
        record = {
            "steps": steps,
            "effective_batch": (
                config["training"]["micro_batch"]
                * config["training"]["grad_accum_steps"]
            ),
            "metrics": metrics,
            "advance_decision": decision,
        }
        records.append(record)
        state["steps"] = records
        write_json_atomic(args.report, state)
        if decision is not None and not decision["advance"]:
            state["status"] = "stopped_at_failed_advance_gate"
            write_json_atomic(args.report, state)
            return
        previous = record
    state["status"] = "complete"
    write_json_atomic(args.report, state)


if __name__ == "__main__":
    main()
