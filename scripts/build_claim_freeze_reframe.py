#!/usr/bin/env python3
"""Freeze the reframed PlanC/B claim from existing frozen reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--todo",
        type=Path,
        default=ROOT / "todo.json",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=ROOT / "outputs/reports/metric_hierarchy_test503_v5.json",
    )
    parser.add_argument(
        "--cad-report",
        type=Path,
        default=ROOT
        / "outputs/reports/final_attribution_no_legalize_test503_v1.json",
    )
    parser.add_argument(
        "--identifiability",
        type=Path,
        default=ROOT
        / "outputs/reports/d0_identifiability_exact_ambiguity_test503_v2.json",
    )
    parser.add_argument(
        "--chip-report",
        type=Path,
        default=ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json",
    )
    parser.add_argument(
        "--house-report",
        type=Path,
        default=ROOT / "outputs/reports/plan_c_house_global_slot_seeds_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/claim_freeze_reframe_v1.json",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "outputs/reports/identifiability_figure_data_v1.json",
    )
    return parser.parse_args()


def number(value: object) -> float:
    return float(value) if value is not None else float("nan")


def main() -> None:
    args = parse_args()
    todo = read(args.todo)
    hierarchy = read(args.hierarchy)["metrics"]
    cad = read(args.cad_report)
    identifiability_path = (
        args.identifiability
        if args.identifiability.is_file()
        else ROOT / "outputs/reports/d0_identifiability_exact_ambiguity_test503.json"
    )
    identifiability = read(identifiability_path)
    chip = read(args.chip_report)
    house = read(args.house_report)

    legal = hierarchy["L0_legal"]
    hard_gate = {
        "fpeval_validity": number(legal.get("fpeval_validity")) >= 0.999,
        "conditional_watertight": number(
            legal.get("conditional_watertight")
        )
        >= 0.98,
        "room_target_recall": number(legal.get("room_target_recall")) >= 0.99,
        "outside_footprint_pixels": number(
            legal.get("outside_footprint_pixels")
        )
        <= 1.0e-6,
    }
    chip_summary = chip.get("official_seed_summary", {})
    house_summary = house.get("summary", {})
    claimable = [
        {
            "claim": (
                "The full target-free pipeline is schema-valid on all 503 test "
                "cases, conditionally watertight on "
                f"{100.0 * legal['conditional_watertight']:.2f}% of "
                "target-watertight cases, reaches "
                f"{100.0 * legal['room_target_recall']:.2f}% target-room "
                "recall, and emits no pixels outside the input footprint."
            ),
            "supported": all(hard_gate.values()),
            "artifact": "metric_hierarchy_test503_v5.json + final_attribution_no_legalize_test503_v1.json",
        },
        {
            "claim": "Structured element/readout composition improves room and instance geometry over raw flow decode without reducing measured object metrics.",
            "supported": (
                cad["room_metrics"]["overlay"]["room_macro_iou"]
                > cad["room_metrics"]["raw"]["room_macro_iou"]
                and all(
                    delta >= 0.0
                    for delta in cad["overlay_object_delta_vs_raw"].values()
                )
            ),
            "artifact": "final_attribution_no_legalize_test503_v1.json",
        },
        {
            "claim": "The correspondence-decoupled conditional token-flow primitive transfers to chip floorplanning across three seeds, with domain-specific decoding and legalization held explicit.",
            "supported": chip.get("gate", {}).get("status") == "pass",
            "artifact": "plan_c_global_slot_seeds_v1.json",
        },
        {
            "claim": "Boundary-to-interior exact reproduction is ill-posed at the 0.95 level.",
            "supported": number(
                identifiability["nearest_train_layout"]["room_macro_iou"][
                    "fraction_ge_095"
                ]
            )
            < 0.05,
            "artifact": str(identifiability_path.relative_to(ROOT)),
        },
        {
            "claim": "The correspondence-decoupling result also transfers to house token-set generation.",
            "supported": house.get("gate", {}).get("status") == "pass",
            "artifact": "plan_c_house_global_slot_seeds_v1.json",
        },
    ]
    report = {
        "status": "pass" if all(item["supported"] for item in claimable) else "fail",
        "reframe": todo["route_decision"],
        "hard_gate": hard_gate,
        "hard_guarantees": legal,
        "weak_metrics": {
            "room_macro_iou": cad["room_metrics"]["overlay"][
                "room_macro_iou"
            ],
            "furniture_iou": cad["object_metrics"]["overlay"][
                "furniture_iou"
            ],
            "fpeval_opening_f1": cad["room_metrics"]["overlay"][
                "fpeval_opening_f1"
            ],
        },
        "identifiability": {
            "strict_0_95_fraction": identifiability[
                "nearest_train_layout"
            ]["room_macro_iou"]["fraction_ge_095"],
            "nearest_train_room_macro_iou_mean": identifiability[
                "nearest_train_layout"
            ]["room_macro_iou"]["mean"],
        },
        "chip_transfer": {
            "gate": chip.get("gate"),
            "summary": chip_summary,
        },
        "house_transfer": {
            "gate": house.get("gate"),
            "summary": house_summary,
        },
        "claimable": claimable,
        "not_claimable": todo["claims"]["not_claimable"],
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    nearest = identifiability["nearest_train_layout"]["room_macro_iou"]
    figure = {
        "source": str(identifiability_path.relative_to(ROOT)),
        "scope": identifiability.get("scope"),
        "summary": nearest,
        "strict_0_95_fraction": nearest["fraction_ge_095"],
        "raw_histogram": identifiability["nearest_train_layout"].get(
            "top1_room_macro_iou_histogram"
        ),
        "per_case_top1_room_macro_iou": identifiability[
            "nearest_train_layout"
        ].get("per_case_top1_room_macro_iou"),
        "interpretation": identifiability.get("interpretation"),
        "best_of_n_used": False,
    }
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_output.write_text(
        json.dumps(figure, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
