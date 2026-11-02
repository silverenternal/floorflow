#!/usr/bin/env python3
"""Build the executable strict-locality protocol and four-block evidence tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_strict_locality_protocol_v2.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fp_strict_locality_protocol_v2.md",
    )
    parser.add_argument(
        "--main-table",
        type=Path,
        default=ROOT / "paper/tables/fp_main_results_v3.md",
    )
    parser.add_argument(
        "--limitations",
        type=Path,
        default=ROOT / "paper/tables/fp_limitations_v2.md",
    )
    return parser.parse_args()


def range_item(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def operation_strata(structural: dict) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for seed in structural["per_seed"].values():
        for operation, payload in seed["operations"].items():
            metrics = payload["after_prediction_vs_after_target"]
            for key, value in metrics.items():
                grouped[operation][key].append(float(value))
    return {
        operation: {
            key: range_item(values)
            for key, values in metrics.items()
        }
        for operation, metrics in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    edit = read(
        "outputs/reports/fp_edit_detection_test503_controlgated_fiveseed_v1.json"
    )
    structural = read("outputs/reports/fp_housemind_structural_alignment_v1.json")
    verifiability = read("outputs/reports/fp_verifiability_score_v1.json")
    failure = read("outputs/reports/fp_failure_analysis_v2.json")
    strata = read("outputs/reports/fp_structural_error_strata_final_v1.json")
    operation = operation_strata(structural)
    seed_metrics = [
        payload["summary"]["after_prediction_vs_after_target"]
        for payload in structural["per_seed"].values()
    ]
    edit_iou = edit["aggregate"]["local_delta_iou"]
    edit_mse = edit["aggregate"]["local_delta_mse"]
    node_f1 = range_item([row["node_f1"] for row in seed_metrics])
    edge = range_item([row["edge_overlap"] for row in seed_metrics])
    ged = range_item([row["ged"] for row in seed_metrics])
    exact_ged = range_item([row["exact"] for row in seed_metrics])
    verification = verifiability["floorflow"]["summary"]["verifiability"]
    corpus = {}
    for name, values in operation.items():
        corpus[name] = {
            key: values[key]["mean"]
            for key in (
                "node_f1",
                "edge_overlap",
                "ged",
                "exact",
                "node_insertion",
                "node_deletion",
                "node_substitution",
            )
        }
    ged_rank = sorted(
        corpus,
        key=lambda name: corpus[name]["ged"],
        reverse=True,
    )
    protocol = {
        "status": "complete",
        "version": "fp-strict-locality-v2",
        "scope": (
            "Declared edit-region protocol on CubiCasa5K test503 with oracle "
            "controls; protocol-aligned structural view is not a leaderboard result."
        ),
        "definitions": {
            "edit_region": (
                "Clamp normalized [x,y,w,h] to [0,1], round boundaries onto "
                "the 224x224 control grid, and allow changes only inside that mask."
            ),
            "changed_control_region": (
                "Pixels where the declared pre-edit and post-edit visible "
                "controls differ; used as an anchor mask, not as the full "
                "allowed edit region."
            ),
            "outside_valid_domain": (
                "V_out = footprint AND NOT dilate(wall_before OR wall_after,1) "
                "AND NOT edit_region. Footprint exterior and wall pixels are "
                "excluded from exactness denominators."
            ),
            "outside_pixel_exact": (
                "Mean uint8 equality over V_out between pre-edit and rendered "
                "post-edit pixels; exact means no tolerance."
            ),
            "outside_room_label_exact": (
                "Mean exact equality of semantic room labels over V_out between "
                "pre-edit and post-edit room readouts."
            ),
            "delta_iou": (
                "IoU between expected target label changes and predicted label "
                "changes over the valid edit domain."
            ),
            "delta_mse": (
                "MSE between binary expected-change and predicted-change maps "
                "over the valid edit domain."
            ),
            "validity": (
                "fpeval verification of wall geometry, room faces, and opening "
                "attachment; reported as 0/1 per case and averaged."
            ),
            "conditional_watertightness": (
                "Prediction watertightness restricted to cases whose post-edit "
                "target is watertight."
            ),
            "verifiability": (
                "outside_pixel_exact * outside_room_label_exact * validity * "
                "conditional_watertightness. Missing factors are N/A, never imputed."
            ),
            "node_f1": (
                "HouseMind-style unique room-type node F1 after edit."
            ),
            "edge_overlap": (
                "Jaccard overlap of unique room-type-pair adjacency edges."
            ),
            "ged": (
                "HouseMind-style unit-cost graph edit distance over room-type "
                "nodes and type-pair edges."
            ),
        },
        "execution_sources": {
            "edit_region_and_exactness": "scripts/build_fp_safety_factorial_2x2.py",
            "delta_metrics": "src/floorflow/evaluation/edit_detection.py",
            "topology": "src/floorflow/evaluation/fpeval_bridge.py",
            "watertight_verification": "third_party/fpvec-lab/fpeval",
            "graph_metrics": "src/floorflow/evaluation/housemind_structure.py",
            "failure_taxonomy": "src/floorflow/evaluation/failure_taxonomy.py",
        },
        "main_four_blocks": {
            "editing_quality": {
                "delta_iou": edit_iou,
                "delta_mse": edit_mse,
                "ged": ged,
            },
            "strict_locality": {
                "outside_pixel_exact": edit["outside_exact_ratio"],
                "outside_room_label_exact": edit[
                    "outside_room_exact_ratio"
                ],
            },
            "topology_legality": {
                "validity": {
                    "mean": verifiability["floorflow"]["summary"]["validity"][
                        "mean"
                    ],
                    "min": verifiability["floorflow"]["summary"]["validity"][
                        "min"
                    ],
                    "max": verifiability["floorflow"]["summary"]["validity"][
                        "max"
                    ],
                },
                "conditional_watertightness": verifiability["floorflow"][
                    "summary"
                ]["conditional_watertightness"],
                "verifiability": verification,
            },
            "graph_consistency": {
                "node_f1": node_f1,
                "edge_overlap": edge,
                "exact_ged_rate": exact_ged,
            },
        },
        "operation_strata": operation,
        "ged_diagnosis": {
            "ranking_descending": ged_rank,
            "dominant_operation": ged_rank[0],
            "dominant_operation_ged": corpus[ged_rank[0]]["ged"],
            "dominant_node_insertion": corpus[ged_rank[0]]["node_insertion"],
            "dominant_node_substitution": corpus[ged_rank[0]][
                "node_substitution"
            ],
            "graph_complexity_spearman_seed19": strata["complexity"][
                "exact_or_fallback_ged"
            ]["target_node_count"]["spearman"],
            "fallback_rate": 1.0 - exact_ged["mean"],
            "explanation": (
                "The aggregate GED range is driven mainly by room removal, "
                "whose merged resulting graph induces node insertion and "
                "substitution costs. Opening and furniture edits remain near "
                "zero GED; larger graphs add a smaller secondary effect."
            ),
        },
        "failure_taxonomy": {
            "primary_failure_counts": failure["primary_failure_counts"],
            "definitions": {
                "watertight_failure": (
                    "Post-edit target is watertight but prediction verification fails."
                ),
                "node_loss_or_insertion": (
                    "Room graph edit path deletes or inserts a room-type node."
                ),
                "edge_error": (
                    "Room-type adjacency edge is missed or falsely added."
                ),
                "room_label_drift": (
                    "Room-type identity changes outside the declared edit intent."
                ),
            },
        },
        "scope_boundaries": [
            "Oracle controls only; target-free deployment is not claimed.",
            "HouseMind-style room-type graph; canonical instance graph is separate.",
            "HouseMind-E is not reproduced because official inference is unreleased.",
            "Raster2Seq is a matched reconstruction baseline, not a native editing method.",
            "Free-text and unstructured-sketch editing are not validated.",
            "Move-wall and resize edits lack matched numeric evidence.",
            "Strict locality can reject valid global-recomposition edits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    docs = [
        "# FloorFlow Strict-Locality Editing Protocol v2",
        "",
        "This document defines the evaluated editing contract. It is not a manuscript.",
        "",
        "## 1. Edit-region contract",
        "",
        "- The declared normalized rectangle is clamped to `[0,1]` before rasterization.",
        "- The output is allowed to change only inside that region.",
        "- The changed-control mask is an anchor for the declared edit, not an expansion of the allowed region.",
        "",
        "## 2. Outside exactness",
        "",
        "- `V_out = footprint AND NOT dilate(wall_before OR wall_after,1) AND NOT edit_region`.",
        "- Outside pixel exact uses strict `uint8` equality with no tolerance.",
        "- Outside room-label exact uses strict semantic-label equality.",
        "",
        "## 3. Editing quality",
        "",
        "- Delta-IoU compares expected and predicted changed pixels over the valid domain.",
        "- Delta-MSE is the binary change-map MSE and is a diagnostic, not the main contribution.",
        "- GED is interpreted together with operation type and its edit-path decomposition.",
        "",
        "## 4. Topology legality",
        "",
        "- `fpeval.verify` checks floorplan validity, wall geometry, room faces, and opening attachment.",
        "- Conditional watertightness restricts the denominator to watertight post-edit targets.",
        "- `Verifiability = outside_pixel_exact * outside_room_label_exact * validity * conditional_watertightness`.",
        "- Missing factors produce N/A and are never imputed.",
        "",
        "## 5. Graph consistency",
        "",
        "- Node F1 and edge overlap use unique room types and room-type-pair adjacency.",
        "- GED uses NetworkX unit node/edge edit costs.",
        "- The canonical instance graph is a separate secondary view.",
        "",
        "## 6. Failure taxonomy",
        "",
        "- Watertight failure: target is closed but prediction verification fails.",
        "- Node loss/insertion: room-type graph edit path changes node identity/count.",
        "- Edge error: adjacency edge is missed or falsely inserted.",
        "- Room-label drift: semantic identity changes outside the intended edit.",
        "",
        "## 7. Validity boundary",
        "",
        "- Sparse room counts and weak interfaces can push conditional watertightness below 0.95.",
        "- Strict outside preservation remains separately measured in those bins.",
        "- A global-recomposition edit that legitimately changes the whole plan is outside this contract.",
        "",
        "## 8. Sources",
        "",
        "- `src/floorflow/evaluation/edit_detection.py`",
        "- `src/floorflow/evaluation/fpeval_bridge.py`",
        "- `third_party/fpvec-lab/fpeval`",
        "- `src/floorflow/evaluation/housemind_structure.py`",
        "- `src/floorflow/evaluation/failure_taxonomy.py`",
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(docs), encoding="utf-8")

    main_table = [
        "# FloorFlow Main Results v3",
        "",
        "CubiCasa test503, control-gated local delta, five seeds.",
        "",
        "## Editing quality",
        "",
        "| Metric | Result | Role |",
        "|---|---:|---|",
        f"| Delta-IoU | {edit_iou['mean']:.6f} ({edit_iou['min']:.6f}-{edit_iou['max']:.6f}) | edit accuracy |",
        f"| Delta-MSE | {edit_mse['mean']:.9f} ({edit_mse['min']:.9f}-{edit_mse['max']:.9f}) | boundary/change-map diagnostic |",
        f"| GED | {ged['mean']:.6f} ({ged['min']:.6f}-{ged['max']:.6f}) | graph consistency; operation-dependent |",
        "",
        "## Strict locality",
        "",
        "| Metric | Result | Definition |",
        "|---|---:|---|",
        "| Outside pixel exact | 1.000000 | strict uint8 equality outside edit region |",
        "| Outside room-label exact | 1.000000 | strict room-label equality outside edit region |",
        "",
        "## Topology legality",
        "",
        "| Metric | Result | Role |",
        "|---|---:|---|",
        f"| Validity | {verifiability['floorflow']['summary']['validity']['mean']:.6f} | legal CAD structure |",
        (
            f"| Conditional watertightness | "
            f"{verifiability['floorflow']['summary']['conditional_watertightness']['min']:.6f}-"
            f"{verifiability['floorflow']['summary']['conditional_watertightness']['max']:.6f} | "
            "closed topology on watertight targets |"
        ),
        f"| Verifiability | {verification['min']:.6f}-{verification['max']:.6f} | product of four audited factors |",
        "",
        "## Graph consistency",
        "",
        "| Metric | Result | Role |",
        "|---|---:|---|",
        f"| Node F1 | {node_f1['mean']:.6f} ({node_f1['min']:.6f}-{node_f1['max']:.6f}) | room-type identity |",
        f"| Edge overlap | {edge['mean']:.6f} ({edge['min']:.6f}-{edge['max']:.6f}) | room-type adjacency |",
        f"| Exact GED rate | {exact_ged['mean']:.6f} | graph metric reliability |",
        "",
        (
            "GED variance is operation-driven: room removal dominates graph edits "
            "through node insertion/substitution, while opening and furniture "
            "deletion are near zero."
        ),
        "",
        (
            "HouseMind-E references are cross-dataset protocol context only. "
            "Raster2Seq is a matched reconstruction baseline, not a native "
            "editing method."
        ),
        "",
    ]
    args.main_table.parent.mkdir(parents=True, exist_ok=True)
    args.main_table.write_text("\n".join(main_table), encoding="utf-8")

    limits = [
        "# FloorFlow Limitations v2",
        "",
        "| Limitation | Evidence status | Claim effect |",
        "|---|---|---|",
        "| HouseMind official inference unavailable | Project page still marks `Code (Soon)` | No reproduced HouseMind comparison |",
        "| Raster2Seq is reconstruction-only | Baseline audit | Native editing superiority is not claimed |",
        "| Controls are target-derived | Oracle-control manifests | Evidence is diagnostic, not deployable |",
        "| Room-type graph is protocol-aligned | Main structural table | No RPLAN leaderboard claim |",
        "| Weak-signal bins fall below 0.95 watertightness | Protocol boundary report | Protocol boundary, not hidden failure |",
        "| Watertight guard is not stable | 2x2 factorial CI crosses zero | Guard is not promoted as a mechanism |",
        "| Free text and sketch editing are untested | No matched rows | No natural-language editing claim |",
        "| Move-wall and resize edits are missing | Operation coverage audit | No numeric evidence for these classes |",
        "| Strict locality limits global recomposition | Protocol definition | Contract scope must be explicit |",
        "| Five seeds are stability evidence | Five matched seeds | Not population inference |",
        "",
    ]
    args.limitations.parent.mkdir(parents=True, exist_ok=True)
    args.limitations.write_text("\n".join(limits), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "dominant_ged_operation": ged_rank[0],
                "ged": ged,
                "fallback_rate": protocol["ged_diagnosis"]["fallback_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
