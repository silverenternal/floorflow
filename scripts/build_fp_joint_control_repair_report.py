#!/usr/bin/env python3
"""Summarize the missing-control joint repair gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

COMPARISON_METRICS = (
    "room_macro_iou",
    "fpeval_wall_f1",
    "fpeval_opening_f1",
    "room_adjacency_f1",
    "conditional_watertight",
    "fpeval_edit_cost",
    "control_room_pixel_preservation",
    "room_support_iou",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("outputs/eval/fp_architecture_ablation_val191_v1/full"),
    )
    parser.add_argument(
        "--nofill-root",
        type=Path,
        default=Path("outputs/eval/fp_robustness_p010_val191_v1"),
    )
    parser.add_argument(
        "--nearest-root",
        type=Path,
        default=Path("outputs/eval/fp_robustness_inpaint_p010_val191_v1"),
    )
    parser.add_argument(
        "--room-repair-root",
        type=Path,
        default=Path("outputs/eval/fp_robustness_roomrepair_p010_val191_v1"),
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=Path("outputs/eval/fp_joint_control_repair_p010_val191_v1"),
    )
    parser.add_argument(
        "--joint-control-manifest",
        type=Path,
        default=Path(
            "outputs/control_contracts/fp_joint_repair_p010_val191_v1/"
            "val/manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_joint_control_repair_v1.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(root: Path) -> dict:
    return {
        "root": str(root),
        "metrics": read_json(root / "metrics.json")["metrics"],
        "per_case": read_json(root / "per_case.json"),
    }


def compare(current: dict, baseline: dict, iterations: int) -> dict:
    metrics = [
        key
        for key in COMPARISON_METRICS
        if key in current["per_case"][0] and key in baseline["per_case"][0]
    ]
    return {
        key: paired_delta(
            current["per_case"],
            baseline["per_case"],
            key,
            iterations=iterations,
        )
        for key in metrics
    }


def main() -> None:
    args = parse_args()
    roots = {
        "clean": args.clean_root,
        "nofill_10": args.nofill_root,
        "nearest_10": args.nearest_root,
        "room_repair_10": args.room_repair_root,
        "joint_repair_10": args.joint_root,
    }
    runs = {name: load_run(root) for name, root in roots.items()}
    clean = runs["clean"]["metrics"]
    joint = runs["joint_repair_10"]["metrics"]
    manifest = read_json(args.joint_control_manifest)
    preservation = manifest["valid_region_preservation"]["minimum"]
    edit_multiplier = joint["fpeval_edit_cost"] / clean["fpeval_edit_cost"]
    adjacency_retention = (
        joint["room_adjacency_f1"] / clean["room_adjacency_f1"]
    )
    comparisons = {
        "nofill_vs_clean": compare(
            runs["nofill_10"],
            runs["clean"],
            args.bootstrap,
        ),
        "nearest_vs_clean": compare(
            runs["nearest_10"],
            runs["clean"],
            args.bootstrap,
        ),
        "room_repair_vs_clean": compare(
            runs["room_repair_10"],
            runs["clean"],
            args.bootstrap,
        ),
        "joint_vs_clean": compare(
            runs["joint_repair_10"],
            runs["clean"],
            args.bootstrap,
        ),
        "joint_vs_nofill": compare(
            runs["joint_repair_10"],
            runs["nofill_10"],
            args.bootstrap,
        ),
        "joint_vs_nearest": compare(
            runs["joint_repair_10"],
            runs["nearest_10"],
            args.bootstrap,
        ),
        "joint_vs_room_repair": compare(
            runs["joint_repair_10"],
            runs["room_repair_10"],
            args.bootstrap,
        ),
    }
    checks = {
        "conditional_watertight_at_least_0_95": (
            joint["conditional_watertight"] >= 0.95
        ),
        "edit_cost_multiplier_below_1_5": edit_multiplier < 1.5,
        "supplied_control_preservation_at_least_0_98": preservation >= 0.98,
        "adjacency_retention_at_least_0_95": adjacency_retention >= 0.95,
        "edit_cost_better_than_nofill": (
            comparisons["joint_vs_nofill"]["fpeval_edit_cost"]["ci_high"]
            < 0.0
        ),
        "adjacency_better_than_nofill": (
            comparisons["joint_vs_nofill"]["room_adjacency_f1"]["ci_low"]
            > 0.0
        ),
        "edit_cost_better_than_nearest": (
            comparisons["joint_vs_nearest"]["fpeval_edit_cost"]["ci_high"]
            < 0.0
        ),
    }
    report = {
        "status": "complete",
        "scope": "CubiCasa val191, 10% spatial control dropout",
        "best_of_n_used": False,
        "control_contract": {
            "source": manifest["source"],
            "target_used_for_control": manifest["target_used_for_control"],
            "deployable": manifest["deployable"],
            "analysis_variant": manifest["analysis_variant"],
            "repair_checkpoint": manifest["repair_checkpoint"],
            "repair_checkpoint_sha256": manifest["repair_checkpoint_sha256"],
            "fallback_non_target_channels": manifest[
                "fallback_non_target_channels"
            ],
        },
        "runs": runs,
        "paired_comparisons": comparisons,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "edit_cost_multiplier": edit_multiplier,
            "adjacency_retention": adjacency_retention,
            "supplied_control_preservation": preservation,
        },
        "decision": (
            "promote joint missing-control repair"
            if all(checks.values())
            else "do not claim missing-control robustness"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
