#!/usr/bin/env python3
"""Launch one frozen chip v2 candidate/control quick-screen command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLAG_OPTIONS = {
    "structured_heads": "--structured-heads",
    "structured_hierarchy_coupling": "--structured-hierarchy-coupling",
    "structured_factorized_block": "--structured-factorized-block",
    "structured_factorized_shared_trunk": "--structured-factorized-shared-trunk",
    "structured_rho_trust_radius": "--structured-rho-trust-radius",
    "structured_dual_ports": "--structured-dual-ports",
    "structured_legalizer_response": "--structured-legalizer-response",
    "structured_legalizer_apply": "--structured-legalizer-apply",
    "structural_relation_head": "--structural-relation-head",
    "relation_specific_attention": "--relation-specific-attention",
    "incidence_normalized_relations": "--incidence-normalized-relations",
    "dynamic_graph_input": "--dynamic-graph-input",
    "complexity_conditioning": "--complexity-conditioning",
    "degree_conditioned_expert": "--degree-conditioned-expert",
    "spatial_route_field": "--spatial-route-field",
    "training_projector": "--training-projector",
    "structural_relation_weight": "--structural-relation-weight",
    "relation_gate_weight": "--relation-gate-weight",
    "coarse_commit_fraction": "--coarse-commit-fraction",
    "coarse_anchor_tolerance": "--coarse-anchor-tolerance",
    "coarse_anchor_strength": "--coarse-anchor-strength",
    "coarse_anchor_max_step": "--coarse-anchor-max-step",
    "training_projector_iterations": "--training-projector-iterations",
    "training_projector_displacement_weight": "--training-projector-displacement-weight",
    "endpoint_compatibility_weight": "--endpoint-compatibility-weight",
    "endpoint_refiner_trajectory_steps": "--endpoint-refiner-trajectory-steps",
    "endpoint_legalized_area_weight": "--endpoint-legalized-area-weight",
    "endpoint_legalized_p2b_weight": "--endpoint-legalized-p2b-weight",
    "endpoint_legalizer_target_weight": "--endpoint-legalizer-target-weight",
    "legalizer_response_weight": "--legalizer-response-weight",
    "legalizer_response_delay_steps": "--legalizer-response-delay-steps",
    "legalizer_response_anneal_steps": "--legalizer-response-anneal-steps",
    "structured_legalizer_strength": "--structured-legalizer-strength",
    "structured_legalizer_max_displacement": "--structured-legalizer-max-displacement",
    "hierarchy_source_mode": "--hierarchy-source-mode",
    "hierarchy_source_levels": "--hierarchy-source-levels",
    "hierarchy_source_shelf_probability": "--hierarchy-source-shelf-probability",
    "hierarchy_source_shelf_probability_final": "--hierarchy-source-shelf-probability-final",
    "hierarchy_source_shelf_probability_steps": "--hierarchy-source-shelf-probability-steps",
    "backbone_unfreeze_step": "--backbone-unfreeze-step",
    "geometry_augment": "--geometry-augment",
    "geometry_augment_probability": "--geometry-augment-probability",
    "routing_proxy_weight": "--routing-proxy-weight",
    "response_sidecar": "--response-sidecar",
    "train_indices": "--train-indices",
}

NEGATABLE = {
    "structured_heads",
    "structured_hierarchy_coupling",
    "structured_factorized_block",
    "structured_dual_ports",
    "structured_legalizer_response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--variant", choices=("candidate", "control"), default="candidate")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tag", default="quick")
    parser.add_argument("--arm-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-shelf-probability", type=float, default=None)
    parser.add_argument("--source-shelf-probability-final", type=float, default=None)
    parser.add_argument("--source-shelf-probability-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flag_args(flags: dict) -> list[str]:
    result = []
    for key, value in flags.items():
        option = FLAG_OPTIONS.get(key)
        if option is None:
            raise KeyError(f"unsupported stage flag: {key}")
        if isinstance(value, bool):
            if value:
                result.append(option)
            elif key in NEGATABLE:
                result.append(option.replace("--", "--no-", 1))
        elif value is not None:
            result.extend((option, str(value)))
    return result


def base_args(
    plan: dict,
    stage: dict,
    variant: str,
    output_dir: Path,
    arm: str,
    device: str,
    batch_size: int,
    steps: int,
    seed: int | None = None,
    data_seed: int | None = None,
) -> list[str]:
    budget = plan["budget"]
    train_indices = stage.get("train_indices", plan["training_subset"])
    training_seed = (
        int(plan["matched_control"]["training_seed"])
        if seed is None
        else int(seed)
    )
    training_data_seed = (
        int(plan["matched_control"]["data_seed"])
        if data_seed is None
        else int(data_seed)
    )
    args = [
        sys.executable,
        "scripts/train_chip_slot.py",
        "--cache",
        "outputs/cache/chip_floorset_50k_train_12k_dev_v2",
        "--arm",
        arm,
        "--architecture",
        "typed_hierarchy",
        "--out-dir",
        str(output_dir),
        "--train-split",
        "train",
        "--val-split",
        "dev",
        "--train-indices",
        train_indices,
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(budget["grad_accum_steps"]),
        "--sampling-mode",
        "random",
        "--gpu-data-cache",
        "--gpu-data-cache-max-gib",
        "16",
        "--eval-batch-size",
        "128",
        "--eval-steps",
        *[str(value) for value in budget["eval_steps"]],
        "--curve-per-count",
        "1",
        "--hidden",
        "192",
        "--global-dim",
        "192",
        "--target-layers",
        "4",
        "--condition-layers",
        "2",
        "--decoder-layers",
        "4",
        "--trinity-depth",
        "12",
        "--hierarchy-depth",
        "12",
        "--typed-layers",
        "1",
        "--heads",
        "3",
        "--head-dim",
        "64",
        "--projector",
        "--projector-iterations",
        "4",
        "--lr",
        "0.0002",
        "--weight-decay",
        "0.001",
        "--warmup",
        "1000",
        "--ema-decay",
        "0.9995",
        "--ema-mode",
        "fixed",
        "--grad-clip",
        "1.0",
        "--grad-clip-per-group",
        "--checkpoint-every",
        str(budget["checkpoint_every"]),
        "--checkpoint-policy",
        "final",
        "--resume",
        plan["parent_checkpoint"],
        "--resume-weights-only",
        "--resume-allow-new-params",
        "--seed",
        str(training_seed),
        "--data-seed",
        str(training_data_seed),
        "--device",
        device,
    ]
    return args


def control_args(include_source_mode: bool = True) -> list[str]:
    result = [
        "--structured-heads",
        "--no-structured-hierarchy-coupling",
        "--no-structured-factorized-block",
        "--structured-dual-ports",
        "--structured-legalizer-response",
    ]
    if include_source_mode:
        result.extend(("--hierarchy-source-mode", "random"))
    return result


def command_for(args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    plan = load(ROOT / "configs/chip_v2_execution_plan_v1.json")
    architecture = load(ROOT / "configs/chip_architecture_completion_v2.json")
    stages = {stage["id"]: stage for stage in plan["stages"]}
    legalizer = plan["legalizer_closed_loop"]
    stages[legalizer["id"]] = legalizer
    if args.stage not in stages:
        raise ValueError(f"unknown stage: {args.stage}")
    stage = stages[args.stage]
    architecture_stages = {item["id"]: item for item in architecture["stages"]}
    architecture_stage = architecture_stages[stage["architecture_stage"]]
    steps = args.steps or plan["budget"]["quick_steps"]
    batch_size = args.batch_size or plan["budget"]["batch_size"]
    variant_arm = "candidate" if args.variant == "candidate" else "control"
    tag = getattr(args, "tag", "quick")
    arm = getattr(args, "arm_name", None) or (
        f"{stage['id'].split('-')[0].lower()}_{variant_arm}_{tag}"
    )
    explicit_output = getattr(args, "output_dir", None)
    output_dir = (
        explicit_output
        if explicit_output is not None
        else (
            ROOT
            / "outputs/chip_hierarchy/v2_authorized_quick"
            / stage["id"].split("-")[0].lower()
            / variant_arm
        )
    )
    command = base_args(
        plan,
        stage,
        args.variant,
        output_dir,
        arm,
        args.device,
        batch_size,
        steps,
        getattr(args, "seed", None),
        getattr(args, "data_seed", None),
    )
    candidate_flags = (
        dict(architecture_stage["flags"])
        if args.variant == "candidate"
        else {}
    )
    source_overrides = {
        "hierarchy_source_shelf_probability": getattr(
            args, "source_shelf_probability", None
        ),
        "hierarchy_source_shelf_probability_final": getattr(
            args, "source_shelf_probability_final", None
        ),
        "hierarchy_source_shelf_probability_steps": getattr(
            args, "source_shelf_probability_steps", None
        ),
    }
    candidate_flags.update(
        {key: value for key, value in source_overrides.items() if value is not None}
    )
    command.extend(control_args(include_source_mode="hierarchy_source_mode" not in candidate_flags))
    if args.variant == "candidate":
        command.extend(flag_args(candidate_flags))
    report = output_dir / f"{arm}.json"
    return command, report, output_dir


def main() -> None:
    args = parse_args()
    command, report, output_dir = command_for(args)
    manifest = {
        "stage": args.stage,
        "variant": args.variant,
        "device": args.device,
        "command": command,
        "output_dir": str(output_dir),
        "report": str(report),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, check=True, env=environment)
    if not report.is_file():
        raise FileNotFoundError(report)


if __name__ == "__main__":
    main()
