#!/usr/bin/env python3
"""Map FloorFlow control sources to their valid quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_evidence_map_v1.json",
    )
    return parser.parse_args()


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256(ROOT / path),
    }


def main() -> None:
    args = parse_args()
    readiness = read_json("outputs/reports/b5_cvpr_readiness.json")
    blocker = read_json("outputs/reports/b1_contour_cfm_blocker.json")
    v11 = read_json("outputs/reports/v11_roomseg_w30_retr0_test503.json")
    oracle_compare = read_json(
        "outputs/reports/fp_completion_oracle_vs_predicted_val191_v1.json"
    )
    predicted_v2 = read_json("outputs/reports/fp_predicted_control_v2.json")
    refiner = read_json("outputs/reports/residential_predicted_refiner_v1.json")
    contour = read_json("outputs/reports/contour_only_contract.json")

    records = [
        {
            "id": "oracle_layout_conditioned",
            "control_source": "target_derived_layout",
            "deployable": False,
            "split": "CubiCasa test503",
            "metrics": {
                "room_iou": readiness["layout_conditioned_results"]
                ["cubicasa_metrics"]["room_macro_iou"],
                "instance_iou": readiness["layout_conditioned_results"]
                ["cubicasa_metrics"]["instance_geometric_iou"],
                "wall_f1": readiness["layout_conditioned_results"]
                ["cubicasa_metrics"]["fpeval_wall_f1"],
                "opening_f1": readiness["layout_conditioned_results"]
                ["cubicasa_metrics"]["fpeval_opening_f1"],
                "validity": readiness["layout_conditioned_results"]
                ["cubicasa_metrics"]["fpeval_validity"],
                "conditional_watertight": readiness[
                    "layout_conditioned_results"
                ]["cubicasa_metrics"]["conditional_watertight"],
            },
            "role": "capacity upper bound",
            "source": source("outputs/reports/b5_cvpr_readiness.json"),
        },
        {
            "id": "target_guided_cfm",
            "control_source": "target_derived_guided_source",
            "deployable": False,
            "split": "CubiCasa validation",
            "metrics": {
                "room_iou": blocker["experiments"]["guided_miTa_cfm"]
                ["resume_best_room_macro_iou"],
            },
            "role": "control-identifiability ceiling",
            "source": source("outputs/reports/b1_contour_cfm_blocker.json"),
        },
        {
            "id": "best_target_free_predicted_control",
            "control_source": "dmd_v11_plus_independent_room_segmenter",
            "deployable": True,
            "split": "CubiCasa test503",
            "metrics": {
                "room_iou": v11["cad_metrics"]["summary"]["room_macro_iou"],
                "instance_iou": v11["cad_metrics"]["summary"]
                ["instance_geometric_iou"],
                "instance_f1": v11["cad_metrics"]["summary"]["instance_room_f1"],
                "type_accuracy": v11["cad_metrics"]["summary"]
                ["instance_type_accuracy"],
                "room_support_iou": v11["cad_metrics"]["summary"]
                ["room_support_iou"],
                "wall_f1": v11["cad_metrics"]["summary"]["fpeval_wall_f1"],
                "opening_f1": v11["cad_metrics"]["summary"]["fpeval_opening_f1"],
                "validity": v11["cad_metrics"]["summary"]["fpeval_validity"],
                "watertight": v11["cad_metrics"]["summary"]["fpeval_watertight"],
            },
            "role": "strongest current target-free evidence",
            "source": source(
                "outputs/reports/v11_roomseg_w30_retr0_test503.json"
            ),
        },
        {
            "id": "predicted_control_completion_v1",
            "control_source": "predicted_v1",
            "deployable": True,
            "split": "CubiCasa val191",
            "metrics": oracle_compare["baseline_metrics"],
            "role": "strict predicted-control baseline",
            "source": source(
                "outputs/reports/fp_completion_oracle_vs_predicted_val191_v1.json"
            ),
        },
        {
            "id": "predicted_control_guided_v2",
            "control_source": "predicted_guided_v1",
            "deployable": True,
            "split": "CubiCasa val64",
            "metrics": predicted_v2["result"],
            "role": "failed broad gate but edit-cost improvement",
            "source": source("outputs/reports/fp_predicted_control_v2.json"),
        },
        {
            "id": "predicted_control_refiner_v1",
            "control_source": "predicted_controls",
            "deployable": True,
            "split": "CubiCasa val491",
            "metrics": {
                **refiner["best"],
                "room_iou_gain": refiner["paired"]["room_macro_iou"][
                    "mean_delta"
                ],
            },
            "role": "target-free local refinement",
            "source": source(
                "outputs/reports/residential_predicted_refiner_v1.json"
            ),
        },
        {
            "id": "contour_only_contract",
            "control_source": "footprint_only",
            "deployable": True,
            "split": "CubiCasa test503",
            "metrics": contour["current_contour_only_metrics"],
            "role": "identifiability reference",
            "source": source("outputs/reports/contour_only_contract.json"),
        },
    ]
    best = records[2]["metrics"]
    report = {
        "schema_version": "predicted-control-evidence-map-v1",
        "scope": "residential_floorplan_only",
        "control_sources_are_not_interchangeable": True,
        "records": records,
        "diagnosis": {
            "oracle_capacity_established": records[0]["metrics"]["room_iou"]
            > 0.95,
            "target_guided_ceiling": records[1]["metrics"]["room_iou"],
            "best_deployable_room_iou": best["room_iou"],
            "best_deployable_instance_iou": best["instance_iou"],
            "best_deployable_support_iou": best["room_support_iou"],
            "identity_gap": best["instance_iou"] - best["room_iou"],
            "interpretation": (
                "The strongest target-free control stack already produces "
                "usable room geometry support and high matched-instance IoU, "
                "but room identity, fragmentation, and class assignment still "
                "suppress class-wise room IoU. The next model work should "
                "convert predicted support into canonical room instances "
                "instead of rebuilding the generator from scratch."
            ),
        },
        "experiment_rules": [
            "Use b5 layout-conditioned numbers only as a capacity upper bound.",
            "Do not treat target-guided CFM as target-free predicted control.",
            "Use v11 as the predicted-control starting point, not the legacy 0.068 test row.",
            "Tune only on validation; the consumed test503 row is evidence, not a development set.",
            "Never compare predicted-control and oracle-control rows as the same condition.",
        ],
        "next_experiment": {
            "task": "PC-02",
            "question": (
                "Can a room-instance identity head convert DMD-v11 predicted "
                "support into coherent room instances while preserving its "
                "instance geometry and validity?"
            ),
            "primary_metrics": [
                "room_macro_iou",
                "instance_room_f1",
                "instance_type_accuracy",
                "fpeval_wall_f1",
                "conditional_watertight",
            ],
            "frozen_baseline": records[2]["metrics"],
        },
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.relative_to(ROOT)),
                "best_deployable_room_iou": best["room_iou"],
                "best_deployable_instance_iou": best["instance_iou"],
                "best_deployable_support_iou": best["room_support_iou"],
                "next_task": report["next_experiment"]["task"],
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
