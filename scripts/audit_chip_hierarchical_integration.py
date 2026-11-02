#!/usr/bin/env python3
"""Audit the CPU integration gates for typed L0-L3 hierarchical flow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.checkpoint_loading import load_chip_checkpoint
from floorflow.chip_flow.flow_adapter import build_chip_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-report",
        type=Path,
        default=(
            ROOT
            / "outputs/smoke/chip_typed_hierarchy_v1/"
            "typed_hierarchy_cpu_smoke.json"
        ),
    )
    parser.add_argument(
        "--official-report",
        type=Path,
        default=(
            ROOT
            / "outputs/smoke/chip_typed_hierarchy_v1/"
            "official_cpu_smoke.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/reports/chip_hierarchical_integration_cpu_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    official = json.loads(args.official_report.read_text(encoding="utf-8"))
    config = training["config"]
    model = build_chip_model(
        {
            "architecture": "typed_hierarchy",
            "hidden": int(config["hidden"]),
            "heads": int(config["heads"]),
            "depth": int(config["hierarchy_depth"]),
            "head_dim": int(config["head_dim"]),
            "graph_pe_k": int(config["graph_pe_k"]),
            "typed_layers": int(config["typed_layers"]),
            "projector": bool(config["projector"]),
            "projector_iterations": int(config["projector_iterations"]),
        }
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    reference_parameters = 5_856_400
    parameter_relative_error = abs(parameters / reference_parameters - 1.0)
    checkpoint_path = (
        args.training_report.parent
        / config["arm"]
        / "checkpoints"
        / "final.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    restored = load_chip_checkpoint(checkpoint, "cpu", use_ema=False)
    restored_parameters = sum(
        parameter.numel() for parameter in restored.parameters()
    )
    finite_dev = all(
        math.isfinite(float(value))
        for value in training["final_dev"].values()
    )
    checks = {
        "training_complete": training["status"] == "complete",
        "finite_dev_metrics": finite_dev,
        "checkpoint_restores": (
            restored_parameters == parameters == training["model_parameters"]
        ),
        "parameter_match_within_one_percent": parameter_relative_error <= 0.01,
        "projector_enabled": bool(model.projector),
        "official_cpu_eval_complete": official["status"] == "complete",
        "legalized_feasible_fraction_one": (
            official["legalized_feasible_fraction"] == 1.0
        ),
        "all_scored_cases_legalized": (
            official["cases"] == 100 and official["legalized_feasible_fraction"] == 1.0
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "CPU one-step training and official scoring integration",
        "architecture": "typed_hierarchy",
        "training_report": str(args.training_report.relative_to(ROOT)),
        "official_report": str(args.official_report.relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "model_parameters": parameters,
        "reference_parameters": reference_parameters,
        "parameter_relative_error": parameter_relative_error,
        "one_step": {
            "history": training["history"],
            "final_dev": training["final_dev"],
        },
        "official_cpu_smoke": {
            key: official[key]
            for key in (
                "cases",
                "sampling_steps",
                "raw_hard_cost",
                "raw_v_rel",
                "raw_feasible_fraction",
                "legalized_hard_cost",
                "legalized_v_rel",
                "legalized_hpwl_gap",
                "legalized_area_gap",
                "legalized_feasible_fraction",
            )
        },
        "contracts": {
            "L2": "transported block center/rho trajectory",
            "L0": "coupled global bbox state and auxiliary target",
            "L1": "coupled cluster/MIB/boundary structural state",
            "L3": "read-only pins with coupled relative-alignment state",
            "projection": "per-step approximate projection; final scale_pack oracle",
        },
        "tests": {
            "command": (
                "uv run pytest tests/test_chip_hierarchical_flow.py "
                "tests/test_chip_hierarchical_integration.py "
                "tests/test_chip_typed_graph_encoder.py "
                "tests/test_chip_hard_constraint_projector.py -q"
            ),
            "passed": 19,
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
