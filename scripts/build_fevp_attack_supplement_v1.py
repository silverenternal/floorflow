#!/usr/bin/env python3
"""Close optional-metric and submetric gaps in the FEVP attack suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.evaluation.edit_detection import edit_detection_iou_mse
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    outside_layer_metrics,
    watertight_diagnostics,
)
from floorflow.evaluation.housemind_structure import (
    RoomTypeGraph,
    housemind_style_ged,
)
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-attacks",
        type=Path,
        default=ROOT / "outputs/reports/fevp_protocol_attack_rows_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_attack_supplement_v1.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_attack_supplement_v1.md",
    )
    return parser.parse_args()


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 2, "Kitchen": 2}),
            "furniture": Counter(),
            "door": Counter(),
            "window": Counter(),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _labels(taxonomy):
    labels = np.zeros((64, 64), dtype=np.uint8)
    wall = taxonomy.class_id("wall")
    bedroom = taxonomy.class_id("room:Bedroom")
    kitchen = taxonomy.class_id("room:Kitchen")
    labels[8:56, 8:31] = bedroom
    labels[8:56, 33:56] = kitchen
    labels[8:56, 8:11] = wall
    labels[8:56, 31:33] = wall
    labels[8:56, 53:56] = wall
    labels[8:11, 8:56] = wall
    labels[53:56, 8:56] = wall
    return labels


def main() -> None:
    args = parse_args()
    base_rows = json.loads(args.base_attacks.read_text(encoding="utf-8"))["rows"]
    supplement = []

    before = np.zeros((16, 16), dtype=np.uint8)
    target_after = before.copy()
    target_after[4:12, 4:12] = 1
    prediction_after = target_after.copy()
    prediction_after[4, 4] = 0
    valid = np.ones_like(before, dtype=bool)
    baseline_iou, baseline_mse = edit_detection_iou_mse(
        before,
        target_after,
        before,
        target_after,
        valid,
    )
    attacked_iou, attacked_mse = edit_detection_iou_mse(
        before,
        target_after,
        before,
        prediction_after,
        valid,
    )
    supplement.extend(
        [
            {
                "metric": "delta_iou",
                "attack": "remove_one_expected_change_pixel",
                "baseline": float(baseline_iou),
                "attacked": float(attacked_iou),
                "sensitive": float(attacked_iou) < float(baseline_iou),
            },
            {
                "metric": "delta_mse",
                "attack": "remove_one_expected_change_pixel",
                "baseline": float(baseline_mse),
                "attacked": float(attacked_mse),
                "sensitive": float(attacked_mse) > float(baseline_mse),
            },
        ]
    )

    taxonomy = _taxonomy()
    labels = _labels(taxonomy)
    plan = raster_labels_to_wall_plan(labels, taxonomy)
    footprint = np.ones_like(labels, dtype=bool)
    baseline_watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=labels,
        footprint=footprint,
        tolerances=FEVPTolerances(),
    )
    larger_hole = labels.copy()
    larger_hole[20:24, 20:24] = 0
    attacked_watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=larger_hole,
        footprint=footprint,
        tolerances=FEVPTolerances(),
    )
    supplement.append(
        {
            "metric": "conditional_watertightness",
            "submetric": "hole_area_ratio",
            "attack": "enlarge_hole_area",
            "baseline": baseline_watertight["hole_area_ratio"],
            "attacked": attacked_watertight["hole_area_ratio"],
            "sensitive": (
                attacked_watertight["hole_area_ratio"]
                > baseline_watertight["hole_area_ratio"]
            ),
        }
    )

    base_map = _labels(taxonomy)
    region = np.zeros_like(base_map, dtype=bool)
    changed = np.zeros_like(region)
    region[4:12, 4:12] = True
    changed[5:9, 5:9] = True
    moved = base_map.copy()
    kitchen = taxonomy.class_id("room:Kitchen")
    moved[8:56, 33:56] = 0
    moved[8:56, 34:57] = kitchen
    baseline_outside = outside_layer_metrics(
        base_map,
        base_map,
        changed,
        region,
        base_map > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    attacked_outside = outside_layer_metrics(
        base_map,
        moved,
        changed,
        region,
        base_map > 0,
        taxonomy,
        tolerances=FEVPTolerances(),
    )
    supplement.append(
        {
            "metric": "outside_exact",
            "submetric": "outside_room_polygon_exact",
            "attack": "move_room_polygon_one_pixel",
            "baseline": baseline_outside["outside_room_polygon_exact"],
            "attacked": attacked_outside["outside_room_polygon_exact"],
            "sensitive": (
                attacked_outside["outside_room_polygon_exact"]
                < baseline_outside["outside_room_polygon_exact"]
            ),
        }
    )
    supplement.append(
        {
            "metric": "outside_exact",
            "submetric": "outside_pixel_exact",
            "attack": "move_room_pixels_outside_edit_region",
            "baseline": baseline_outside["outside_pixel_exact"],
            "attacked": attacked_outside["outside_pixel_exact"],
            "sensitive": (
                attacked_outside["outside_pixel_exact"]
                < baseline_outside["outside_pixel_exact"]
            ),
        }
    )

    target_graph = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Kitchen"}),
        edges=frozenset({("Bedroom", "Kitchen")}),
    )
    attacked_graph = RoomTypeGraph(
        labels=frozenset({"Bedroom", "Bath"}),
        edges=frozenset({("Bedroom", "Bath")}),
    )
    supplement.append(
        {
            "metric": "ged",
            "submetric": "ged",
            "attack": "relabel_one_graph_node",
            "baseline": housemind_style_ged(
                target_graph,
                target_graph,
            )["ged"],
            "attacked": housemind_style_ged(
                attacked_graph,
                target_graph,
            )["ged"],
            "sensitive": housemind_style_ged(
                attacked_graph,
                target_graph,
            )["ged"]
            > 0.0,
        }
    )

    covered_pairs = sorted(
        {
            (
                row["metric"],
                row.get("submetric", row["metric"]),
            )
            for row in [*base_rows, *supplement]
        }
    )
    required_pairs = [
        ("outside_exact", "outside_pixel_exact"),
        ("outside_exact", "outside_room_level_exact"),
        ("outside_exact", "outside_room_label_exact"),
        ("outside_exact", "outside_room_polygon_exact"),
        ("outside_exact", "outside_edge_exact"),
        ("outside_exact", "locality_violation_rate"),
        ("validity", "outer_boundary_closed"),
        ("validity", "room_polygon_closed"),
        ("validity", "no_self_intersection"),
        ("validity", "no_room_overlap"),
        ("validity", "semantic_complete"),
        ("conditional_watertightness", "watertight_rate"),
        ("conditional_watertightness", "hole_count"),
        ("conditional_watertightness", "hole_area_ratio"),
        ("conditional_watertightness", "disconnected_component_count"),
        ("conditional_watertightness", "dangling_endpoint_rate"),
        ("node_f1", "precision"),
        ("node_f1", "recall"),
        ("node_f1", "f1"),
        ("edge_overlap", "precision"),
        ("edge_overlap", "recall"),
        ("edge_overlap", "f1"),
        ("delta_iou", "delta_iou"),
        ("delta_mse", "delta_mse"),
        ("ged", "ged"),
    ]
    report = {
        "status": "complete",
        "base_attacks": len(base_rows),
        "supplemental_attacks": len(supplement),
        "attacks": supplement,
        "covered_metric_submetric_pairs": covered_pairs,
        "required_metric_submetric_pairs": required_pairs,
        "checks": {
            "all_supplemental_attacks_sensitive": all(
                row["sensitive"] for row in supplement
            ),
            "mandatory_and_optional_submetrics_covered": all(
                item in covered_pairs for item in required_pairs
            ),
            "delta_metrics_attacked": all(
                any(row["metric"] == metric for row in supplement)
                for metric in ("delta_iou", "delta_mse")
            ),
            "hole_area_attacked": any(
                row.get("submetric") == "hole_area_ratio" for row in supplement
            ),
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# FEVP Attack Coverage Supplement",
        "",
        "| Metric | Submetric | Attack | Baseline | Attacked | Sensitive |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in supplement:
        table.append(
            f"| {row['metric']} | {row.get('submetric', '—')} | "
            f"{row['attack']} | {row['baseline']:.6f} | "
            f"{row['attacked']:.6f} | {row['sensitive']} |"
        )
    table.extend(
        [
            "",
            f"Covered metric/submetric pairs: {len(covered_pairs)}.",
            (
                "Required metric/submetric pairs covered: "
                f"{all(item in covered_pairs for item in required_pairs)}."
            ),
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
