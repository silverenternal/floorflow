#!/usr/bin/env python3
"""Build reviewer risks for the locality-auditability-gap claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v7.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v10.json").read_text(
            encoding="utf-8"
        )
    )
    final = evidence["locality_auditability_evidence"]["final_claim"]
    house = evidence["locality_auditability_evidence"]["house_mind"]
    pareto = evidence["locality_auditability_evidence"]["pareto_v3"]
    audit_map = evidence["locality_auditability_evidence"]["audit_map"]
    deletion = evidence["locality_auditability_evidence"]["room_deletion"]
    risks = [
        {
            "risk": "FEVP is presented as another metric bundle",
            "evidence": "The claim starts from the missing auditability of editing locality.",
            "response": "Lead with the auditability problem, then the protocol.",
            "residual": "The protocol is still new and lacks broad external adoption.",
        },
        {
            "risk": "HouseMind is ignored or attacked",
            "evidence": "HouseMind-E/O are acknowledged with public editing metrics.",
            "response": "Use complementary framing and never claim to beat HouseMind.",
            "residual": "Official HouseMind inference is not released.",
        },
        {
            "risk": "HouseMind is claimed to lack locality",
            "evidence": house["non_ranking_statement"],
            "response": "State only that the public evaluation does not expose the audit.",
            "residual": "Qualitative localized editing remains a valid HouseMind claim.",
        },
        {
            "risk": "Unreported audit cells are treated as failures",
            "evidence": audit_map["footnote"],
            "response": "Keep the dash definition beside the evidence map.",
            "residual": "Readers may still interpret missing cells negatively.",
        },
        {
            "risk": "HouseMind-G is treated as an editing baseline",
            "evidence": "HouseMind-G is generation-only and excluded from the editing axis.",
            "response": "Label G as generation-only wherever HouseMind variants appear.",
            "residual": "Its generation results cannot support editing conclusions.",
        },
        {
            "risk": "HouseMind is assigned a fake locality point",
            "evidence": "HouseMind-E/G/O publish no outside exact or violation rate.",
            "response": "Use a context card, not an invented x-coordinate.",
            "residual": "No cross-paper locality plot is possible.",
        },
        {
            "risk": "Typed routing only wins locality",
            "evidence": (
                "With corrected evaluation, Typed is better than No graph on "
                "Delta-IoU, Node F1, Edge F1, and GED."
            ),
            "response": "Use the corrected v3 Pareto and disclose the frontier-calculation fix.",
            "residual": "Typed remains a post-hoc matched routing mechanism.",
        },
        {
            "risk": "No-graph is unfairly omitted",
            "evidence": "No-graph remains in the Pareto with zero violation and lower quality.",
            "response": "Show all four audited arms.",
            "residual": "No-graph may be preferable under another objective weighting.",
        },
        {
            "risk": "The evidence map overstates missing evidence",
            "evidence": "Dashes state not reported/not comparable, not failed.",
            "response": "Use neutral symbols and the required footnote.",
            "residual": "Different evaluation units remain difficult for readers.",
        },
        {
            "risk": "Room-deletion weakness is hidden",
            "evidence": (
                "Room deletion GED is 5.418 and contributes "
                f"{deletion['ged_tail']['by_operation']['room_remove']['tail_share']:.4f} "
                "of the top GED decile, while outside exact remains 1.0."
            ),
            "response": "Present deletion as a topology-difficulty mode, not a locality failure.",
            "residual": "Deletion remains an unsolved hard mode.",
        },
        {
            "risk": "Validity or watertightness is promoted as a net gain",
            "evidence": "Five-check validity is 0.97975; guard has 57 repairs and 67 regressions.",
            "response": "Keep both as safety baselines only.",
            "residual": "Hardened validity remains protocol-dependent.",
        },
        {
            "risk": "External methods enter a merged leaderboard",
            "evidence": "RPLAN/CubiCasa task, data, and case units differ.",
            "response": "Keep external values as separate context and never rank on missing cells.",
            "residual": "No matched external editing-aware baseline exists.",
        },
        {
            "risk": "Human preference is missing",
            "evidence": "Human evaluation is excluded by scope.",
            "response": "Make no subjective-quality claim.",
            "residual": "Perceived design quality remains unmeasured.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v7",
        "",
        "Locality-auditability-gap evidence only.",
        "",
        "| Risk | Evidence | Response | Residual |",
        "|---|---|---|---|",
    ]
    for risk in risks:
        lines.append(
            f"| {risk['risk']} | {risk['evidence']} | "
            f"{risk['response']} | {risk['residual']} |"
        )
    lines.extend(
        [
            "",
            "## Final Claim",
            "",
            f"> {final['claim_en']}",
            "",
            "## Required HouseMind Sentence",
            "",
            f"> {house['non_ranking_statement']}",
            "",
            "## Forbidden",
            "",
            *[f"- {claim}" for claim in final["forbidden_claims"]],
            "",
            "## Evidence Pointers",
            "",
            "- HouseMind: `outputs/reports/fevp_housemind_positioning_v2.json`.",
            "- Pareto v3: `outputs/reports/fevp_mechanism_pareto_v3.json`.",
            "- Audit map: `outputs/reports/fevp_audit_evidence_map_v2.json`.",
            "- Room-deletion diagnosis: `outputs/reports/fevp_room_deletion_diagnosis_v2.json`.",
            "",
        ]
    )
    _ = pareto
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
