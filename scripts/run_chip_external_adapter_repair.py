#!/usr/bin/env python3
"""Screen adapter/legalizer repairs on frozen development cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorflow.chip_flow.refiner_config import append_refiner_protocol
from scripts.run_chip_def_adapter_suite import case_manifest, load_json

DEFAULT_CONFIG = ROOT / "configs/chip_external_adapter_v1.json"
DEFAULT_MANIFEST = ROOT / "outputs/reports/chip_external_benchmark_manifest_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/chip_external_adapter_repair_v1.json"

SCREENS = {
    "expanded": {
        "legalizer_relation": "expanded",
        "instance_hierarchy": False,
        "preserve_reference_orientation": False,
    },
    "hierarchy": {
        "legalizer_relation": "restored",
        "instance_hierarchy": True,
        "preserve_reference_orientation": False,
    },
    "expanded_hierarchy_orientation": {
        "legalizer_relation": "expanded",
        "instance_hierarchy": True,
        "preserve_reference_orientation": True,
    },
    "strict_route_match_repair": {
        "legalizer_relation": "expanded",
        "instance_hierarchy": True,
        "preserve_reference_orientation": True,
        "force_shelf_cases": ["bp_be", "bp_multi"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--screens", nargs="+", choices=tuple(SCREENS), default=["expanded"])
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def relative(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control in (None, 0):
        return None
    return float((candidate - control) / control)


def report_path(
    screen: str,
    nickname: str,
    arm: str,
    seed: int,
    policy: dict[str, Any],
) -> Path:
    root = ROOT / "outputs/reports/chip_external_adapter_repair"
    if policy.get("protocol_tag"):
        root = root / str(policy["protocol_tag"])
    return (
        root
        / screen
        / nickname
        / f"{arm}_seed{seed}.json"
    )


def run_one(
    args: argparse.Namespace,
    config: dict[str, Any],
    manifest: dict[str, Any],
    design_key: str,
    arm: str,
    screen: str,
) -> dict[str, Any]:
    case = {**case_manifest(manifest, design_key), **config["cases"][design_key]}
    spec = SCREENS[screen]
    adapter_arm = f"repair_{screen}_{arm}"
    output = report_path(
        screen,
        case["design_nickname"],
        arm,
        args.seed,
        config["primary_protocol"],
    )
    if not output.is_file() or load_json(output).get("status") != "pass":
        command = [
            sys.executable,
            str(ROOT / "scripts/run_chip_external_placement.py"),
            "--case",
            case["design_nickname"],
            "--checkpoint",
            str(ROOT / config[arm]["seeds"][str(args.seed)]),
            "--arm",
            adapter_arm,
            "--seed",
            str(args.seed),
            "--odb",
            str(ROOT / case["input_odb"]),
            "--device",
            args.device,
            "--nfe",
            str(config["primary_protocol"]["nfe"]),
            "--refiner-steps",
            str(config["primary_protocol"]["refiner_steps"]),
            "--projector-iterations",
            str(config["primary_protocol"]["candidate_projector_iterations"]),
            "--macro-halo-um",
            str(case["macro_halo_um"]),
            "--legalizer-relation",
            str(spec["legalizer_relation"]),
            "--output",
            str(output),
        ]
        append_refiner_protocol(command, config["primary_protocol"])
        if spec["instance_hierarchy"]:
            command.append("--instance-hierarchy")
        if spec["preserve_reference_orientation"]:
            command.append("--preserve-reference-orientation")
        if case["design_nickname"] in spec.get("force_shelf_cases", []):
            command.append("--force-shelf")
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "case": design_key,
                "arm": arm,
                "screen": screen,
                "status": "failed",
                "command": command,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
    return {"case": design_key, "arm": arm, "screen": screen, "status": "completed"}


def summarize(
    screen: str,
    cases: list[str],
    seed: int,
    manifest: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    pairs = []
    for design_key in cases:
        nickname = case_manifest(manifest, design_key)["design_nickname"]
        candidate_path = report_path(
            screen,
            nickname,
            "candidate",
            seed,
            policy,
        )
        control_path = report_path(screen, nickname, "control", seed, policy)
        if not candidate_path.is_file() or not control_path.is_file():
            pairs.append({"case": design_key, "status": "missing"})
            continue
        candidate = load_json(candidate_path)
        control = load_json(control_path)
        candidate_conn = candidate.get("connectivity", {}).get("legalized_output", {})
        control_conn = control.get("connectivity", {}).get("legalized_output", {})
        candidate_check = candidate.get("geometry", {}).get("legalized_output", {})
        control_check = control.get("geometry", {}).get("legalized_output", {})
        pairs.append(
            {
                "case": design_key,
                "status": (
                    "completed"
                    if candidate.get("status") == "pass"
                    and control.get("status") == "pass"
                    else "failed"
                ),
                "routes": {
                    "candidate": candidate.get("legalizer_route"),
                    "control": control.get("legalizer_route"),
                    "matched": (
                        candidate.get("legalizer_route")
                        == control.get("legalizer_route")
                    ),
                },
                "legalized_feasible": {
                    "candidate": candidate_check.get("checker_feasible"),
                    "control": control_check.get("checker_feasible"),
                },
                "overlap_pairs": {
                    "candidate": candidate_check.get("overlap_pair_count"),
                    "control": control_check.get("overlap_pair_count"),
                },
                "b2b_hpwl_relative": relative(
                    candidate_conn.get("b2b_hpwl_um"),
                    control_conn.get("b2b_hpwl_um"),
                ),
                "p2b_hpwl_relative": relative(
                    candidate_conn.get("p2b_hpwl_um"),
                    control_conn.get("p2b_hpwl_um"),
                ),
            }
        )
    completed = [pair for pair in pairs if pair["status"] == "completed"]
    candidate_routes = [pair["routes"]["candidate"] for pair in completed]
    control_routes = [pair["routes"]["control"] for pair in completed]
    return {
        "pairs": pairs,
        "summary": {
            "completed_pairs": len(completed),
            "candidate_scale_pack_count": sum(
                str(route).startswith("scale_pack") for route in candidate_routes
            ),
            "control_scale_pack_count": sum(
                str(route).startswith("scale_pack") for route in control_routes
            ),
            "matched_route_pairs": sum(
                bool(pair["routes"]["matched"]) for pair in completed
            ),
            "all_legalized_feasible": all(
                pair["legalized_feasible"]["candidate"]
                and pair["legalized_feasible"]["control"]
                for pair in completed
            ),
            "all_zero_overlap": all(
                pair["overlap_pairs"]["candidate"] == 0
                and pair["overlap_pairs"]["control"] == 0
                for pair in completed
            ),
        },
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    cases = manifest["selection"]["selected_cases"]
    runs = []
    for screen in args.screens:
        for design_key in cases:
            for arm in ("candidate", "control"):
                runs.append(
                    run_one(args, config, manifest, design_key, arm, screen)
                )
    summaries = {
        screen: summarize(
            screen,
            cases,
            args.seed,
            manifest,
            config["primary_protocol"],
        )
        for screen in args.screens
    }
    all_pairs_completed = all(
        summary["summary"]["completed_pairs"] == len(cases)
        and summary["summary"]["all_legalized_feasible"]
        and summary["summary"]["all_zero_overlap"]
        for summary in summaries.values()
    )
    report = {
        "version": "2026-10-28-chip-external-adapter-repair-v1",
        "status": "screen_complete"
        if all(run["status"] == "completed" for run in runs)
        and all_pairs_completed
        else "screen_incomplete",
        "seed": args.seed,
        "cases": cases,
        "protocol": {
            "checkpoint_selection": "frozen checkpoints",
            "nfe": config["primary_protocol"]["nfe"],
            "refiner_steps": config["primary_protocol"]["refiner_steps"],
            "projector_iterations": config["primary_protocol"][
                "candidate_projector_iterations"
            ],
            "one_variable_rule": (
                "Run expanded alone before combining it with hierarchy or orientation."
            ),
        },
        "screens": {
            screen: SCREENS[screen]
            for screen in args.screens
        },
        "runs": runs,
        "results": summaries,
        "selection_policy": (
            "Do not select by downstream PPA. Select the least confounded policy that "
            "restores legality and route matching, then freeze it before holdout evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": summaries}, indent=2))
    if report["status"] != "screen_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
