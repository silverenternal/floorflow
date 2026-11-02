#!/usr/bin/env python3
"""Quantify raster-to-vector consistency for the frozen candidate."""

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
        "--hierarchy",
        type=Path,
        default=ROOT / "outputs/reports/metric_hierarchy_test503_v5.json",
    )
    parser.add_argument(
        "--cad-report",
        type=Path,
        default=ROOT / "outputs/reports/final_gated_targetfree_cad_test503_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster_vector_consistency_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy = read(args.hierarchy)
    metrics = hierarchy["metrics"]
    cad = read(args.cad_report)
    overlay = cad["room_metrics"]["overlay"]
    objects = cad["object_metrics"]["overlay"]
    vector = {
        "fpeval_validity": overlay["fpeval_validity"],
        "fpeval_watertight": overlay["fpeval_watertight"],
        "fpeval_wall_f1": overlay["fpeval_wall_f1"],
        "fpeval_opening_f1": overlay["fpeval_opening_f1"],
        "fpeval_room_face_f1": overlay["fpeval_room_face_f1"],
        "fpeval_room_polygon_f1": overlay["fpeval_room_polygon_f1"],
        "fpeval_edit_cost": overlay["fpeval_edit_cost"],
    }
    raster = {
        "wall_iou": objects["wall_iou"],
        "opening_iou": objects["opening_iou"],
        "furniture_iou": objects["furniture_iou"],
        "opening_f1": objects["opening_f1"],
    }
    topology = {
        "wall_graph_f1": metrics["L1_topology"]["wall_graph_f1"],
        "opening_wall_alignment": metrics["L1_topology"][
            "opening_wall_alignment"
        ],
        "furniture_room_containment": metrics["L1_topology"][
            "furniture_room_containment"
        ],
        "room_component_gap": metrics["L1_topology"]["room_component_gap"],
    }
    report = {
        "status": "audit",
        "scope": "Frozen contour-conditioned candidate on CubiCasa test503.",
        "raster": raster,
        "vector": vector,
        "topology": topology,
        "consistency_gaps": {
            "wall_f1_to_iou": float(vector["fpeval_wall_f1"]) - float(raster["wall_iou"]),
            "opening_f1_to_iou": float(vector["fpeval_opening_f1"]) - float(raster["opening_iou"]),
            "room_polygon_f1": vector["fpeval_room_polygon_f1"],
        },
        "module_ownership": hierarchy.get("module_ownership"),
        "interpretation": (
            "Vectorization is a separate, attributable stage after raster "
            "generation; exact CAD reproduction is not claimed."
        ),
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
