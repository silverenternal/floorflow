#!/usr/bin/env python3
"""Compare the best per-layer checkpoint with the flat serial composition."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    opening = _load("predicted_control_centerwidth_ranker_nms20_val491_v1.json")
    wall = _load("predicted_control_wallgap002_val491_v1.json")
    flat = _load("predicted_control_combined_wallopen_val491_v1.json")
    interaction = _load(
        "predicted_control_wallgap_interaction_val491_paired_v1.json"
    )
    opening_summary = opening["summary"]
    wall_summary = wall["summary"]
    flat_summary = flat["summary"]
    opening_delta = interaction["paired"]["fpeval_opening_f1"]
    report = {
        "schema_version": "predicted-stack-serial-layer-v1",
        "scope": "residential CubiCasa predicted-control val491",
        "test_opened": False,
        "target_used_at_inference": False,
        "layers": {
            "wall": {
                "checkpoint_contract": "reconcile_gap_fraction=0.02",
                "wall_f1": wall_summary["fpeval_wall_f1_mean"],
                "wall_precision": wall_summary["fpeval_wall_precision_mean"],
                "wall_recall": wall_summary["fpeval_wall_recall_mean"],
                "continuity": wall_summary["wall_continuity_rate_mean"],
            },
            "opening": {
                "checkpoint_contract": "center-width + HGB verifier + NMS20",
                "opening_f1": opening_summary["fpeval_opening_f1_mean"],
                "room_iou": opening_summary["room_macro_iou_mean"],
            },
        },
        "flat_serial_composition": {
            "opening_f1": flat_summary["fpeval_opening_f1_mean"],
            "wall_f1": flat_summary["fpeval_wall_f1_mean"],
            "room_iou": flat_summary["room_macro_iou_mean"],
            "node_f1": flat_summary["node_f1_mean"],
            "edge_overlap": flat_summary["edge_overlap_mean"],
            "ged": flat_summary["ged_mean"],
            "validity": flat_summary["validity_mean"],
            "functional_watertight": flat_summary[
                "functional_watertight_mean"
            ],
        },
        "interaction": {
            "opening_delta_vs_opening_only": opening_delta["mean_delta"],
            "opening_ci95": [opening_delta["ci_low"], opening_delta["ci_high"]],
            "wall_delta_vs_wall_only": 0.0,
        },
        "decision": "retain_layer_specific_evidence; do_not_use_flat_geometry_for_claims",
        "sources": {
            "opening": "outputs/reports/predicted_control_centerwidth_ranker_nms20_val491_v1.json",
            "wall": "outputs/reports/predicted_control_wallgap002_val491_v1.json",
            "flat": "outputs/reports/predicted_control_combined_wallopen_val491_v1.json",
            "interaction": "outputs/reports/predicted_control_wallgap_interaction_val491_paired_v1.json",
        },
    }
    output = REPORTS / "predicted_control_stack_serial_layer01_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
