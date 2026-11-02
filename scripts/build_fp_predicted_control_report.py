#!/usr/bin/env python3
"""Gate the target-free predicted-control v2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("outputs/reports/fp_predicted_control_v2_comparison_val64.json"),
    )
    parser.add_argument(
        "--current-control-manifest",
        type=Path,
        default=Path(
            "outputs/control_contracts/fp_predicted_guided_v1_s1_r224_val64/"
            "val/manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/fp_predicted_control_v2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    manifest = json.loads(
        args.current_control_manifest.read_text(encoding="utf-8")
    )
    current = comparison["current_metrics"]
    paired = comparison["paired"]
    checks = {
        "edit_cost_better_than_v1": (
            paired["fpeval_edit_cost"]["ci_high"] < 0.0
        ),
        "type_accuracy_at_least_0_10": (
            current["instance_type_accuracy"] >= 0.10
        ),
        "validity_at_least_0_99": current["fpeval_validity"] >= 0.99,
        "conditional_watertight_at_least_0_95": (
            current["conditional_watertight"] >= 0.95
        ),
        "wall_f1_not_worse": (
            paired["fpeval_wall_f1"]["ci_low"] >= 0.0
        ),
        "door_contact_not_worse": (
            paired["door_contact_room_fraction"]["ci_low"] >= 0.0
        ),
    }
    report = {
        "status": "failed_gate",
        "scope": "CubiCasa val64 target-free predicted-control development split",
        "target_used_for_control": manifest["target_used_for_control"],
        "deployable": manifest["deployable"],
        "best_of_n_used": False,
        "current_primitives": {
            "program": "room_program_set_full_v3",
            "room_layout": "guided_room_cfm_full_v1, one guided step",
            "control_builder": "contour_controls.predict_contour_controls",
        },
        "comparison": str(args.comparison),
        "current_control_manifest": str(args.current_control_manifest),
        "checks": checks,
        "passed": all(checks.values()),
        "result": {
            "edit_cost": current["fpeval_edit_cost"],
            "edit_cost_reduction_vs_v1": -paired["fpeval_edit_cost"][
                "mean_delta"
            ],
            "room_macro_iou": current["room_macro_iou"],
            "instance_type_accuracy": current["instance_type_accuracy"],
            "conditional_watertight": current["conditional_watertight"],
            "door_contact_room_fraction": current[
                "door_contact_room_fraction"
            ],
            "room_components": current["room_components"],
        },
        "decision": (
            "Do not claim end-to-end automation. The v2 path lowers edit cost "
            "relative to predicted-v1 but remains far outside a usable CAD "
            "distribution and violates the type, wall, and topology gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **checks}, indent=2))


if __name__ == "__main__":
    main()
