from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_chip_v2_quick_plan.py"
    spec = importlib.util.spec_from_file_location("chip_v2_quick_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(stage: str, variant: str) -> Namespace:
    return Namespace(
        stage=stage,
        variant=variant,
        steps=20,
        device="cuda:0",
        batch_size=4,
        tag="quick",
        output_dir=None,
        dry_run=True,
    )


def test_m50_control_disables_its_mechanisms() -> None:
    command, _, _ = _module().command_for(
        _args("M50-COMMITTED-HIERARCHY-FLOW", "control")
    )
    assert "--no-structured-hierarchy-coupling" in command
    assert "--no-structured-factorized-block" in command
    assert "--structural-relation-head" not in command
    assert "--coarse-commit-fraction" not in command


def test_m53_candidate_keeps_legalizer_action_for_m54() -> None:
    module = _module()
    candidate, _, _ = module.command_for(
        _args("M53-REFINER-COMPATIBLE-FLOW", "candidate")
    )
    legalizer, _, _ = module.command_for(
        _args("M54-LEGALIZATION-AWARE-FLOW", "candidate")
    )
    assert "--endpoint-compatibility-weight" in candidate
    assert "--structured-legalizer-apply" not in candidate
    assert "--structured-legalizer-apply" in legalizer
    assert "--response-sidecar" in legalizer


def test_confirmation_tag_keeps_exploratory_outputs_immutable() -> None:
    args = _args("M64-SOURCE-ONLY-CLEAN", "candidate")
    args.steps = 5000
    args.tag = "confirmation5k"
    args.output_dir = ROOT / "outputs/chip_hierarchy/m64_confirmation_5k_v1/candidate"
    command, report, output_dir = _module().command_for(args)

    assert output_dir == args.output_dir
    assert report.name == "m64_candidate_confirmation5k.json"
    assert command[command.index("--steps") + 1] == "5000"
    assert "--hierarchy-source-mode" in command
    assert "mixed" in command


def test_source_schedule_override_replaces_stage_defaults() -> None:
    args = _args("M64-SOURCE-ONLY-CLEAN", "candidate")
    args.source_shelf_probability = 0.5
    args.source_shelf_probability_final = 0.5
    args.source_shelf_probability_steps = 1
    command, _, _ = _module().command_for(args)

    positions = [
        index
        for index, value in enumerate(command)
        if value == "--hierarchy-source-shelf-probability"
    ]
    assert command[positions[-1] + 1] == "0.5"
    final_positions = [
        index
        for index, value in enumerate(command)
        if value == "--hierarchy-source-shelf-probability-final"
    ]
    assert command[final_positions[-1] + 1] == "0.5"


def test_training_seed_override_uses_last_argument() -> None:
    args = _args("M64-SOURCE-ONLY-CLEAN", "control")
    args.seed = 19
    args.data_seed = 19
    command, _, _ = _module().command_for(args)

    seed_positions = [
        index for index, value in enumerate(command) if value == "--seed"
    ]
    data_seed_positions = [
        index for index, value in enumerate(command) if value == "--data-seed"
    ]
    assert command[seed_positions[-1] + 1] == "19"
    assert command[data_seed_positions[-1] + 1] == "19"
