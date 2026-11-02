#!/usr/bin/env python3
"""Replicate the high-block range contrast on seeds 20 and 21."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.training_runtime import write_json_atomic
from scripts.run_chip_v46_response_queue import run_logged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21])
    parser.add_argument(
        "--range-config",
        type=Path,
        default=ROOT / "configs/chip_sota_training_range_compute_ladder_v1.json",
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=ROOT / "configs/chip_sota_training_subset_control_v2.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchy/range_multiseed_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/chip_range_multiseed_v1.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clone_config(
    base: dict,
    seed: int,
    label: str,
    *,
    reference_eval: Path | None = None,
    device: str = "cuda:0",
    eval_device: str = "cuda:1",
) -> dict:
    config = json.loads(json.dumps(base))
    config["status"] = f"frozen_multiseed_{label}_seed{seed}"
    config["training"]["seed"] = int(seed)
    config["steps"] = [20000]
    config["device"] = device
    config["evaluation"]["device"] = eval_device
    if reference_eval is not None:
        config["reference_eval"] = str(reference_eval)
    return config


def train_command(config: Path, output_root: Path, report: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_chip_sota_training_ladder.py"),
        "--config",
        str(config),
        "--output-root",
        str(output_root),
        "--report",
        str(report),
    ]


def comparison_command(
    control_eval: Path,
    range_eval: Path,
    output: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_chip_training_frontier_comparison.py"),
        "--baseline",
        str(control_eval),
        "--candidate",
        str(range_eval),
        "--output",
        str(output),
        "--scope",
        f"seed{seed} dev32: 21-120 frozen subset vs clean 21-60 frozen subset",
    ]


def main() -> None:
    args = parse_args()
    args.range_config = args.range_config.resolve()
    args.control_config = args.control_config.resolve()
    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    range_base = load(args.range_config)
    control_base = load(args.control_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_root = args.report.parent / args.report.stem / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    state = {
        "version": "chip-range-multiseed-replication-v1",
        "status": "running",
        "seeds": args.seeds,
        "runs": [],
        "comparisons": [],
    }
    write_json_atomic(args.report, state)
    started = time.perf_counter()
    for seed in args.seeds:
        seed_root = args.output_root / f"seed{seed}"
        control_config = config_root / f"control_seed{seed}.json"
        range_config = config_root / f"range_seed{seed}.json"
        write_json_atomic(
            control_config,
            clone_config(
                control_base,
                seed,
                "control_21_60",
                device=args.device,
                eval_device=args.eval_device,
            ),
        )
        write_json_atomic(
            range_config,
            clone_config(
                range_base,
                seed,
                "range_21_120",
                reference_eval=seed_root / "control/eval_20000.json",
                device=args.device,
                eval_device=args.eval_device,
            ),
        )
        control_report = args.report.parent / f"chip_range_control_seed{seed}.json"
        range_report = args.report.parent / f"chip_range_candidate_seed{seed}.json"
        for name, config, out_root, report in (
            ("control", control_config, seed_root / "control", control_report),
            ("range", range_config, seed_root / "range", range_report),
        ):
            result = run_logged(
                train_command(config, out_root, report),
                args.report.parent / f"chip_range_{name}_seed{seed}.log",
            )
            state["runs"].append(
                {"seed": seed, "name": name, "config": str(config), **result}
            )
            write_json_atomic(args.report, state)
            if result["exit_code"] != 0:
                state["status"] = "failed_training"
                write_json_atomic(args.report, state)
                raise SystemExit(1)
        control_eval = seed_root / "control/eval_20000.json"
        range_eval = seed_root / "range/eval_20000.json"
        comparison = (
            args.report.parent / f"chip_range_comparison_seed{seed}.json"
        )
        result = run_logged(
            comparison_command(control_eval, range_eval, comparison, seed),
            args.report.parent / f"chip_range_comparison_seed{seed}.log",
        )
        state["runs"].append(
            {"seed": seed, "name": "comparison", **result}
        )
        if result["exit_code"] == 0 and comparison.is_file():
            payload = load(comparison)
            state["comparisons"].append(
                {
                    "seed": seed,
                    "status": payload["status"],
                    "cost_relative": payload["paired"][
                        "legalized_hard_cost"
                    ]["relative_delta"],
                    "v_rel_relative": payload["paired"][
                        "legalized_v_rel"
                    ]["relative_delta"],
                    "cost_ci": [
                        payload["paired"]["legalized_hard_cost"]["ci_low"],
                        payload["paired"]["legalized_hard_cost"]["ci_high"],
                    ],
                    "v_rel_ci": [
                        payload["paired"]["legalized_v_rel"]["ci_low"],
                        payload["paired"]["legalized_v_rel"]["ci_high"],
                    ],
                    "report": str(comparison.relative_to(ROOT)),
                }
            )
        write_json_atomic(args.report, state)
    state["elapsed_seconds"] = time.perf_counter() - started
    state["status"] = "complete"
    write_json_atomic(args.report, state)
    print(json.dumps(state))


if __name__ == "__main__":
    main()
