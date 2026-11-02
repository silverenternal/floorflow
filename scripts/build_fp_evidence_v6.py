#!/usr/bin/env python3
"""Build the strict-locality protocol evidence ledger v6."""

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
        default=ROOT / "outputs/reports/fp_cvpr_evidence_v6.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = copy.deepcopy(read("outputs/reports/fp_cvpr_evidence_v5.json"))
    house = read("outputs/reports/fp_housemind_public_status_v2.json")
    protocol = read("outputs/reports/fp_strict_locality_protocol_v2.json")
    typed = read("outputs/reports/fp_typed_edit_graph_ablation_v1.json")
    strata = read("outputs/reports/fp_edit_type_strata_v2.json")
    pareto = read("outputs/reports/fp_dual_pareto_v2.json")
    protocol_claim = (
        "FloorFlow promotes edit locality from a soft expectation to an "
        "auditable protocol constraint: changes are admitted only inside the "
        "declared edit region, outside pixel and room-label exactness are "
        "measured directly, topology legality and structural graph consistency "
        "remain separately auditable, and missing verifiability factors are "
        "reported as N/A."
    )
    evidence.update(
        {
            "status": "complete",
            "claim_status": (
                "strict_locality_protocol_and_verifiable_evidence_suite_complete"
            ),
            "primary_narrative": protocol_claim,
            "protocol_claim": protocol_claim,
            "released_claim": (
                evidence["released_claim"]
                + " The primary contribution is the strict-locality protocol "
                "and its verifiable evidence suite, not a leaderboard SOTA claim."
            ),
            "housemind_public_status": house,
            "protocol_v2": protocol,
            "typed_edit_graph_ablation": typed,
            "edit_type_strata_v2": strata,
            "dual_pareto_v2": pareto,
            "submission_status": (
                "strict_locality_protocol_evidence_ready_for_scoped_cvpr_submission"
            ),
            "experiment_objective_complete": True,
            "package_target": "outputs/cvpr_package_fp_v11",
        }
    )
    evidence["claims"]["supported_protocol_v2"] = [
        protocol_claim,
        (
            "HouseMind's official project page still marks code as `Code (Soon)` "
            "while exposing a runnable demo; no matched strict-locality result "
            "is claimed."
        ),
        (
            "The typed edit graph preserves outside exactness 1.0 and improves "
            "Delta-IoU and graph consistency over dense and graphless routing, "
            "with a small non-significant watertightness point-estimate trade-off."
        ),
        (
            "Room removal explains the GED tail through edge and node graph edits; "
            "the aggregate range is not an unexplained instability."
        ),
        (
            "Wall-move, resize, and sequential multi-step edits remain unavailable "
            "and receive no numeric claim."
        ),
    ]
    evidence["claims"]["not_supported"] = list(
        dict.fromkeys(
            evidence["claims"]["not_supported"]
            + [
                "HouseMind complete-method superiority or a same-protocol ranking.",
                "A first-method novelty claim.",
                "Numeric claims for wall-move, room-resize, or multi-step edits.",
                "The typed graph as a native retrained cross-attention mechanism.",
                "FloorSet or complete CAD SOTA.",
            ]
        )
    )
    new_gates = {
        "housemind_public_status_verified": house["status"] == "verified",
        "housemind_code_unreleased_disclosed": (
            house["public_artifacts"]["code_status"] == "code_soon"
            and not house["public_artifacts"]["official_code_available"]
        ),
        "strict_locality_protocol_complete": protocol["status"] == "complete",
        "typed_graph_ablation_complete": typed["status"] == "complete",
        "typed_graph_supported_with_tradeoff": (
            typed["hypothesis_status"] == "supported_with_topology_tradeoff"
        ),
        "edit_type_strata_complete": strata["status"] == "complete",
        "edit_type_checks_complete": all(strata["checks"].values()),
        "dual_pareto_complete": pareto["status"] == "complete",
        "dual_pareto_checks_complete": all(pareto["checks"].values()),
    }
    evidence["gates"].update(new_gates)
    evidence["new_gate_checks"] = new_gates
    evidence["blocking_work"] = list(
        dict.fromkeys(
            evidence["blocking_work"]
            + [
                "Typed graph routing is post-hoc over matched rendered outputs, not retrained cross-attention.",
                "Wall-move, room-resize, and multi-step edits lack matched rows.",
                "The safety-efficiency panel measures routing latency with fixed model NFE.",
                "HouseMind's official inference pipeline remains unavailable.",
            ]
        )
    )
    evidence["protocol_sources"] = {
        "definition": "outputs/reports/fp_strict_locality_protocol_v2.json",
        "main_table": "paper/tables/fp_main_results_v3.md",
        "typed_graph": "outputs/reports/fp_typed_edit_graph_ablation_v1.json",
        "edit_types": "outputs/reports/fp_edit_type_strata_v2.json",
        "pareto": "outputs/reports/fp_dual_pareto_v2.json",
        "housemind": "outputs/reports/fp_housemind_public_status_v2.json",
    }
    failures = [name for name, passed in new_gates.items() if not passed]
    if failures:
        raise ValueError(f"floorplan v6 gates failed: {failures}")
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


if __name__ == "__main__":
    main()
