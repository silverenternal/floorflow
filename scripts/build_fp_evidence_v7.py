#!/usr/bin/env python3
"""Integrate FEVP and external-audit evidence into the final experiment ledger."""

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
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v7.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(read("outputs/reports/fp_cvpr_evidence_v6.json"))
    coverage = read("outputs/reports/fevp_coverage_matrix_v1.json")
    mechanism = read("outputs/reports/fp_locality_mechanism_v2.json")
    raster2seq = read("outputs/reports/fp_raster2seq_official_anchor_v1.json")
    raster2seq_adapter = read(
        "outputs/reports/fp_raster2seq_identity_reconstruction_v1.json"
    )
    ce2eplan = read("outputs/reports/ce2eplan_official_anchor_v1.json")
    ce2eplan_availability = read("outputs/reports/ce2eplan_availability_v1.json")
    ce2eplan_gate = read(
        "outputs/reports/ce2eplan_floorflow_matching_gate_v1.json"
    )
    program = read("fptodo.json")
    primary_claim = (
        "Under FEVP v1.0, FloorFlow is the only method in the evaluated "
        "coverage matrix with a complete report for all five mandatory "
        "strict-locality metrics, while retaining high edit and graph quality."
    )
    evidence.update(
        {
            "status": "complete",
            "claim_status": "fevp_protocol_and_external_audits_complete",
            "primary_narrative": primary_claim,
            "released_claim": (
                f"{evidence['released_claim']} The final experiment package "
                "frames locality as an auditable protocol constraint and "
                "separates matched, unreported, and cross-dataset evidence."
            ),
            "fevp_v1": {
                "protocol": "docs/fevp_v1_0.md",
                "coverage": coverage,
                "mechanism": mechanism,
                "verifiability_definition": (
                    "outside_pixel_exact * outside_room_label_exact * "
                    "validity * conditional_watertightness"
                ),
            },
            "external_audits": {
                "Raster2Seq": {
                    "official_anchor": raster2seq,
                    "adapter_diagnostic": raster2seq_adapter,
                },
                "CE2EPlan": {
                    "official_anchor": ce2eplan,
                    "availability": ce2eplan_availability,
                    "matching_gate": ce2eplan_gate,
                },
            },
            "submission_status": (
                "cvpr_experiment_evidence_ready_for_scoped_protocol_submission"
            ),
            "experiment_objective_complete": True,
            "package_target": "outputs/cvpr_package_fp_v12",
        }
    )
    evidence["claims"]["supported_fevp_v1"] = [
        primary_claim,
        (
            "Outside exactness, validity, conditional watertightness, Node F1, "
            "and edge overlap are reported as separate mandatory dimensions."
        ),
        (
            "The matched routing audit separates locality from quality: the "
            "control gate preserves exact locality, while structural routing "
            "recovers edit and graph quality."
        ),
        (
            "Raster2Seq's official RoomF1 reproduces the public CubiCasa anchor "
            "within 0.002; its case unit cannot be bijected to all 503 "
            "FloorFlow SVG floors, so no matched identity rank is claimed."
        ),
        (
            "CE2EPlan's released RPLAN mode-auto checkpoint reproduces FID near "
            "the published value and all six paper-style statistic ratios; "
            "graph-edit diagnostics remain independent because evaluator code "
            "is not public."
        ),
        "HouseMind remains a complementary MLLM framework with code marked Code (Soon).",
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"]
            + [
                "A 503-case matched Raster2Seq identity rank.",
                "The Raster2Seq adapter diagnostic as an official score.",
                "CE2EPlan as a matched editing baseline.",
                "A merged RPLAN/CubiCasa external leaderboard.",
                "Exact reproduction of CE2EPlan's undisclosed GED evaluator.",
                "Any human-preference or participant-derived result.",
            ]
        )
    )
    new_gates = {
        "fevp_schema_validator_complete": (
            Path(ROOT / "schema/fevp_report_v1.schema.json").is_file()
        ),
        "fevp_coverage_complete": coverage["status"] == "complete",
        "fevp_floorflow_only_complete_row": (
            coverage["claims"]["floorflow_complete_under_fevp"]
        ),
        "fevp_no_unreported_ranking": (
            coverage["claims"]["external_unreported_cells_are_not_failures"]
        ),
        "locality_mechanism_complete": mechanism["status"] == "complete",
        "locality_gate_mechanism": mechanism["checks"][
            "no_gate_violates_contract"
        ],
        "raster2seq_official_anchor_complete": (
            raster2seq["status"]
            == "official_anchor_complete_matched_identity_blocked"
        ),
        "raster2seq_identity_mislabel_prevented": (
            raster2seq_adapter["claim_gate"]["matched_table_eligible"] is False
        ),
        "ce2eplan_official_anchor_complete": (
            ce2eplan["status"]
            == "official_rplan_anchor_reproduced_with_independent_diagnostics"
        ),
        "ce2eplan_not_matched": (
            ce2eplan_gate["matched_generation_comparison_available"] is False
        ),
        "human_evaluation_excluded": program["program_contract"][
            "human_evaluation"
        ].startswith("excluded"),
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["blocking_work"] = list(
        dict.fromkeys(
            evidence["blocking_work"]
            + [
                (
                    "Raster2Seq contour-region cases require an authoritative "
                    "region-to-floor manifest before a matched 503 identity table."
                ),
                "CE2EPlan has no license and has not been matched on RPLAN.",
                (
                    "CE2EPlan GED is an independent diagnostic because its "
                    "exact evaluator implementation is not released."
                ),
                "HouseMind official inference code remains unavailable.",
            ]
        )
    )
    evidence["protocol_sources"].update(
        {
            "fevp_definition": "docs/fevp_v1_0.md",
            "fevp_coverage": "outputs/reports/fevp_coverage_matrix_v1.json",
            "fevp_mechanism": "outputs/reports/fp_locality_mechanism_v2.json",
            "raster2seq_official": (
                "outputs/reports/fp_raster2seq_official_anchor_v1.json"
            ),
            "raster2seq_adapter": (
                "outputs/reports/fp_raster2seq_identity_reconstruction_v1.json"
            ),
            "ce2eplan": "outputs/reports/ce2eplan_official_anchor_v1.json",
            "ce2eplan_gate": (
                "outputs/reports/ce2eplan_floorflow_matching_gate_v1.json"
            ),
        }
    )
    failures = [name for name, passed in new_gates.items() if not passed]
    if failures:
        raise ValueError(f"floorplan v7 gates failed: {failures}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "new_gates": len(new_gates),
                "failed": failures,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
