#!/usr/bin/env python3
"""Attribute floorplan legality and quality across deterministic decode stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-legalize",
        type=Path,
        default=ROOT
        / "outputs/reports/final_attribution_no_legalize_test503_v1.json",
    )
    parser.add_argument(
        "--legalize",
        type=Path,
        default=ROOT
        / "outputs/reports/final_legalized_fixed_attribution_test503_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/constraint_attribution_test503_v1.json",
    )
    return parser.parse_args()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path) -> dict | None:
    return read(path) if path.is_file() else None


def stage_metrics(
    room: dict,
    objects: dict,
    constraints: dict,
    *,
    conditional_watertight: float | None,
) -> dict:
    keys = (
        "room_macro_iou",
        "instance_geometric_iou",
        "instance_room_f1",
        "room_component_gap",
        "fpeval_validity",
        "fpeval_watertight",
        "fpeval_wall_f1",
        "fpeval_opening_f1",
    )
    result = {key: room.get(key) for key in keys}
    result.update({key: objects.get(key) for key in objects})
    result.update(constraints)
    result["conditional_watertight"] = conditional_watertight
    return result


def main() -> None:
    args = parse_args()
    no_legalize = read(args.no_legalize)
    legalize = read(args.legalize)
    if no_legalize["room_metrics"]["raw"] != legalize["room_metrics"]["raw"]:
        raise ValueError("raw flow metrics differ; ablation is not paired")
    base_constraints = no_legalize["constraint_metrics"]
    legal_constraints = legalize["constraint_metrics"]
    stages = {
        "raw_flow": stage_metrics(
            no_legalize["room_metrics"]["raw"],
            no_legalize["object_metrics"]["raw"],
            {},
            conditional_watertight=None,
        ),
        "element_room_readout": stage_metrics(
            no_legalize["room_metrics"]["element_room"],
            no_legalize["object_metrics"]["raw"],
            {},
            conditional_watertight=None,
        ),
        "overlay_no_legalization": stage_metrics(
            no_legalize["room_metrics"]["overlay"],
            no_legalize["object_metrics"]["overlay"],
            base_constraints,
            conditional_watertight=no_legalize["cad_metrics"]["summary"][
                "conditional_watertight"
            ],
        ),
        "overlay_with_legalization": stage_metrics(
            legalize["room_metrics"]["overlay"],
            legalize["object_metrics"]["overlay"],
            legal_constraints,
            conditional_watertight=legalize["cad_metrics"]["summary"][
                "conditional_watertight"
            ],
        ),
    }
    keys = (
        "room_macro_iou",
        "instance_geometric_iou",
        "fpeval_validity",
        "fpeval_watertight",
        "conditional_watertight",
        "outside_footprint_pixels",
        "room_target_recall",
        "opening_iou",
        "opening_wall_alignment",
        "furniture_iou",
        "room_component_gap",
    )
    delta = {
        key: stages["overlay_with_legalization"].get(key, 0.0)
        - stages["overlay_no_legalization"].get(key, 0.0)
        for key in keys
    }
    legalization_safe = (
        delta["conditional_watertight"] >= 0.0
        and delta["fpeval_watertight"] >= 0.0
        and delta["opening_iou"] >= 0.0
    )
    sweep = {}
    for name in ("noop", "r2", "r4", "r6"):
        path = ROOT / f"outputs/reports/legalization_val64_{name}_v2.json"
        result = optional(path)
        if result is None:
            continue
        summary = result["cad_metrics"]["summary"]
        sweep[name] = {
            "conditional_watertight": summary["conditional_watertight"],
            "fpeval_watertight": summary["fpeval_watertight"],
            "opening_iou": summary["opening_iou"],
            "opening_wall_alignment": summary["opening_wall_alignment"],
            "room_macro_iou": summary["room_macro_iou"],
        }
    fixed_val = optional(
        ROOT / "outputs/reports/legalization_val64_fixed_v3.json"
    )
    if fixed_val is not None:
        summary = fixed_val["cad_metrics"]["summary"]
        sweep["fixed_r2"] = {
            "conditional_watertight": summary["conditional_watertight"],
            "fpeval_watertight": summary["fpeval_watertight"],
            "opening_iou": summary["opening_iou"],
            "opening_wall_alignment": summary["opening_wall_alignment"],
            "room_macro_iou": summary["room_macro_iou"],
        }
    report = {
        "status": "complete",
        "decision": (
            "keep_legalization_safe_no_regression"
            if legalization_safe
            else "reject_legalization_for_frozen_pipeline"
        ),
        "protocol": {
            "split": "CubiCasa test",
            "cases": no_legalize["cases"],
            "checkpoint": no_legalize["checkpoint"],
            "paired_decode_command": True,
            "change": "only --legalize-objects toggled",
            "best_of_n_used": False,
            "target_derived_control": False,
        },
        "stages": stages,
        "legalization_delta": delta,
        "checks": {
            "raw_flow_identical": True,
            "legalizer_preserves_conditional_watertight": (
                delta["conditional_watertight"] >= 0.0
            ),
            "legalizer_preserves_unconditional_watertight": (
                delta["fpeval_watertight"] >= 0.0
            ),
            "legalizer_preserves_opening_quality": (
                delta["opening_iou"] >= 0.0
            ),
        },
        "val64_radius_sweep": sweep,
        "regression_history": {
            "pre_fix_test503_conditional_watertight": 0.13513513513513511,
            "pre_fix_test503_opening_iou": 0.07924362884036132,
            "root_cause": (
                "Wall-proximity filtering deleted valid opening labels; "
                "underlying wall pixels were then exposed during composition."
            ),
            "fix": (
                "Legalization now preserves footprint-valid openings and "
                "filters only impossible furniture unless closure is requested."
            ),
        },
        "scope": (
            "Flow-state footprint masking is invariant inside the trained model "
            "and is not toggled here. The audit separates raw flow, structured "
            "room readout, overlay composition, and optional legalization."
        ),
        "sources": [
            str(args.no_legalize.relative_to(ROOT)),
            str(args.legalize.relative_to(ROOT)),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
