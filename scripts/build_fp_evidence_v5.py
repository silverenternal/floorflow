#!/usr/bin/env python3
"""Promote the completed floorplan protocol evidence from v4 to v5."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v5.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(read("outputs/reports/fp_cvpr_evidence_v4.json"))
    verifiability = read("outputs/reports/fp_verifiability_score_v1.json")
    pareto = read("outputs/reports/fp_pareto_matched_v1.json")
    factorial = read("outputs/reports/fp_safety_factorial_2x2_v1.json")
    boundary = read("outputs/reports/fp_protocol_boundary_v1.json")
    structural = read("outputs/reports/fp_housemind_structural_alignment_v1.json")
    source_map = read("outputs/reports/fp_main_claim_source_map_v1.json")
    cvpr = read("outputs/reports/fp_cvpr2026_statistics_v1.json")
    related = read("outputs/reports/fp_strict_locality_related_work_v1.json")
    edit = evidence["key_numbers"]["edit_detection"]
    iou = edit["aggregate"]["local_delta_iou"]
    mse = edit["aggregate"]["local_delta_mse"]
    watertight = verifiability["floorflow"]["summary"][
        "conditional_watertightness"
    ]
    after = [
        seed["summary"]["after_prediction_vs_after_target"]
        for seed in structural["per_seed"].values()
    ]
    node_f1 = {
        "min": min(row["node_f1"] for row in after),
        "max": max(row["node_f1"] for row in after),
    }
    edge_overlap = {
        "min": min(row["edge_overlap"] for row in after),
        "max": max(row["edge_overlap"] for row in after),
    }
    ged = {
        "min": min(row["ged"] for row in after),
        "max": max(row["ged"] for row in after),
    }
    main_claim = (
        "On CubiCasa test503 with oracle controls and a HouseMind-style "
        "room-type graph, FloorFlow's control-gated local-delta output "
        f"reaches Delta-IoU {iou['min']:.4f}-{iou['max']:.4f}, "
        f"Delta-MSE {mse['min']:.6f}-{mse['max']:.6f}, strict outside "
        "pixel and room-label exactness 1.0, validity 1.0, and conditional "
        f"watertightness {watertight['min']:.4f}-{watertight['max']:.4f} across "
        f"{len(edit['seeds'])} seeds. Under the aligned after-edit graph, "
        f"Node F1 is {node_f1['min']:.4f}-{node_f1['max']:.4f}, edge overlap "
        f"{edge_overlap['min']:.4f}-{edge_overlap['max']:.4f}, and GED "
        f"{ged['min']:.4f}-{ged['max']:.4f}."
    )
    evidence.update(
        {
            "status": "complete",
            "scope": (
                "CubiCasa5K test503 control-gated protocol plus matched "
                "floorplan evidence; house-mind-style structure uses oracle controls"
            ),
            "released_claim": main_claim,
            "main_claim": main_claim,
            "claim_status": (
                "scoped_cvpr_experiment_evidence_complete_with_disclosed_boundaries"
            ),
            "required_boundary_sentence": (
                "FloorFlow does not claim to beat HouseMind-E's complete "
                "natural-language editing method and does not claim complete "
                "CAD-generation SOTA."
            ),
            "primary_claim_evidence": {
                "edit_metrics": edit,
                "verifiability": verifiability,
                "housemind_style_structure": structural,
                "matched_pareto": pareto,
                "safety_factorial": factorial,
                "protocol_boundary": boundary,
                "claim_source_map": source_map,
            },
            "positioning": {
                "strict_locality": related,
                "cvpr_statistics": cvpr,
                "allowed": (
                    "Edit safety is an independent evaluation axis; FloorFlow "
                    "reports an unusually complete auditable combination."
                ),
                "not_claimed": [
                    "Priority or first-method novelty",
                    "HouseMind-E complete-method superiority",
                    "HouseMind RPLAN leaderboard rank",
                    "RPLAN, FloorSet, or complete CAD SOTA",
                ],
            },
            "submission_ready": True,
            "submission_status": (
                "experiment_evidence_ready_for_scoped_cvpr_submission"
            ),
            "experiment_objective_complete": True,
        }
    )
    evidence["claims"]["supported_main_protocol"] = [
        main_claim,
        (
            "Verifiability, defined as the product of outside pixel exact, "
            "outside room-label exact, validity, and conditional watertightness, "
            f"is {verifiability['floorflow']['summary']['verifiability']['min']:.4f}-"
            f"{verifiability['floorflow']['summary']['verifiability']['max']:.4f} "
            "across five seeds; missing external factors remain N/A."
        ),
        (
            "FloorFlow control-gated is the only frontier point in both matched "
            "edit-quality/locality panels; HouseMind-E is kept as a separate "
            "cross-dataset context anchor."
        ),
        (
            "The 2x2 matched audit identifies the control gate as the locality "
            "mechanism. The watertight guard has no stable aggregate gain."
        ),
        (
            "Weak room-count and weak-interface bins define the current protocol "
            "boundary: watertightness can fall below 0.95 while strict locality "
            "remains 1.0."
        ),
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"]
            + [
                "The watertight guard improves aggregate watertightness.",
                "Structural metrics are deployable target-free results.",
                "A cross-dataset HouseMind-E point belongs to the matched Pareto frontier.",
                "Matched reconstruction evidence is a native editing-method comparison.",
            ]
        )
    )
    new_gates = {
        "verifiability_factor_complete": all(verifiability["checks"].values()),
        "matched_pareto_complete": all(pareto["checks"].values()),
        "safety_factorial_complete": all(factorial["checks"].values()),
        "control_gate_is_locality_mechanism": (
            factorial["effects"]["control_gate_on_minus_off"][
                "outside_pixel_exact"
            ]["mean_delta"]
            > 0.0
        ),
        "watertight_guard_not_promoted": (
            factorial["effects"]["guard_on_minus_off"][
                "conditional_watertight"
            ]["ci_low"]
            < 0.0
            < factorial["effects"]["guard_on_minus_off"][
                "conditional_watertight"
            ]["ci_high"]
        ),
        "protocol_boundary_complete": boundary["status"] == "complete",
        "cvpr_statistics_verified": cvpr["status"] == "verified",
        "strict_locality_related_work_complete": related["status"]
        == "complete",
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["blocking_work"] = [
        "HouseMind official inference code remains unreleased.",
        "No native editing-aware external method was run under the matched protocol.",
        "Raster2Seq is a reconstruction baseline; its outside-frozen wrapper is a weak editing baseline.",
        "HouseMind-style structural metrics are oracle-control and cross-dataset aligned, not a HouseMind leaderboard rank.",
        "Conditional watertightness falls below 0.95 in disclosed weak-signal bins.",
        "The watertight guard is not retained as an aggregate improvement mechanism.",
        "The priority claim remains downgraded; no first-method claim is allowed.",
    ]
    evidence["evidence_v5"] = {
        "status": "complete",
        "promotions": [
            "HouseMind-style after-edit structure moved to the main structural table.",
            "Verifiability reported factor-by-factor and as a composite.",
            "Matched Pareto and 2x2 safety factorial added.",
            "Protocol boundaries and CVPR-scale positioning recorded.",
        ],
        "package_target": "outputs/cvpr_package_fp_v10",
    }
    evidence["source_artifacts"] = sorted(
        {
            *evidence.get("source_artifacts", []),
            "outputs/reports/fp_verifiability_score_v1.json",
            "outputs/reports/fp_pareto_matched_v1.json",
            "outputs/reports/fp_safety_factorial_2x2_v1.json",
            "outputs/reports/fp_protocol_boundary_v1.json",
            "outputs/reports/fp_housemind_structural_alignment_v1.json",
            "outputs/reports/fp_main_claim_source_map_v1.json",
            "outputs/reports/fp_cvpr2026_statistics_v1.json",
            "outputs/reports/fp_strict_locality_related_work_v1.json",
        }
    )
    failures = [name for name, passed in new_gates.items() if not passed]
    if failures:
        raise ValueError(f"floorplan v5 gates failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "gates": len(evidence["gates"]),
                "failed": failures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
