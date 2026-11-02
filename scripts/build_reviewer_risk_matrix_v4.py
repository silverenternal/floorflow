#!/usr/bin/env python3
"""Build the final FEVP-aware reviewer-risk matrix."""

from __future__ import annotations

import argparse
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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v4.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = read("outputs/reports/fp_cvpr_evidence_v7.json")
    coverage = evidence["fevp_v1"]["coverage"]
    mechanism = evidence["fevp_v1"]["mechanism"]
    raster2seq = evidence["external_audits"]["Raster2Seq"]
    ce2eplan = evidence["external_audits"]["CE2EPlan"]
    r2s_audit = raster2seq["official_anchor"]["case_unit_audit"]
    r2s_anchor = raster2seq["official_anchor"]["official_anchor"]["metrics"]
    ce2e = ce2eplan["official_anchor"]["reproduced"]
    risks = [
        {
            "risk": "FEVP is dismissed as a self-defined benchmark",
            "evidence": (
                "FEVP separates five mandatory auditable metrics and rejects "
                "imputation; the task need is outside preservation for editing."
            ),
            "response": (
                "Position locality as a deployment requirement and publish the "
                "schema, reference implementation, rows, and validator."
            ),
            "residual": (
                "No external editing-aware method has yet produced a complete "
                "FEVP row."
            ),
        },
        {
            "risk": "Missing external cells are treated as failures",
            "evidence": (
                "The coverage matrix defines dashes as not reported or not "
                "comparable, never as zero or inferior."
            ),
            "response": (
                "Keep the N/A definition beside every matrix/table and avoid "
                "ranking rows with missing factors."
            ),
            "residual": "Readers may still misread a coverage matrix as a rank.",
        },
        {
            "risk": "HouseMind is treated as a like-for-like baseline",
            "evidence": (
                "HouseMind is a complementary MLLM framework and its official "
                "code remains Code (Soon)."
            ),
            "response": (
                "Describe only protocol coverage and never claim complete-method "
                "or same-benchmark superiority."
            ),
            "residual": "Official inference and token serialization are unavailable.",
        },
        {
            "risk": "Raster2Seq is mischaracterized as editing-aware",
            "evidence": "Raster2Seq is a CubiCasa reconstruction method.",
            "response": (
                "Call it a matched reconstruction baseline or separate official "
                "reconstruction anchor."
            ),
            "residual": "No native editing-aware external method is reproduced.",
        },
        {
            "risk": "Raster2Seq identity cases are falsely described as matched",
            "evidence": (
                f"Official has {r2s_audit['official_annotation_cases']} "
                f"contour-region cases versus {r2s_audit['project_floor_cases']} "
                "FloorFlow floor cases; "
                f"{len(r2s_audit['mismatched_sources'])} sources differ."
            ),
            "response": (
                "Report the official RoomF1 anchor separately and mark the 503 "
                "identity bijection blocked."
            ),
            "residual": (
                "A matched table requires an authoritative region-to-floor manifest."
            ),
        },
        {
            "risk": "The Raster2Seq adapter diagnostic is presented as official",
            "evidence": (
                "The 503-row adapter uses a project-defined official-style renderer."
            ),
            "response": (
                "Keep it appendix-only and explicitly ineligible for matched tables."
            ),
            "residual": "It remains useful only as renderer-sensitivity evidence.",
        },
        {
            "risk": "Raster2Seq official reproduction is overclaimed",
            "evidence": (
                f"Reproduced RoomF1 is {r2s_anchor['room_f1']:.6f} versus "
                "the public 0.887 anchor."
            ),
            "response": (
                "State that the task anchor reproduces within 0.002, not that "
                "FloorFlow beats or matches the upstream pipeline."
            ),
            "residual": "Exact upstream environment/version details remain external.",
        },
        {
            "risk": "CE2EPlan is treated as a matched editing baseline",
            "evidence": (
                "CE2EPlan performs RPLAN boundary-conditioned generation; "
                "FloorFlow evidence is CubiCasa local editing."
            ),
            "response": "Use CE2EPlan only as a cross-dataset context anchor.",
            "residual": "No FloorFlow RPLAN checkpoint or matched run exists.",
        },
        {
            "risk": "CE2EPlan diagnostics are claimed as exact paper reproduction",
            "evidence": (
                f"Reproduced FID {ce2e['FID']:.4f} is close to 1.08, but the "
                f"independent GED {ce2e['GED_mean']:.4f} differs from 3.48."
            ),
            "response": (
                "Report FID/statistics as reproduction evidence and GED as an "
                "independent diagnostic with an unpublished evaluator gap."
            ),
            "residual": "The exact CE2EPlan GED implementation is not released.",
        },
        {
            "risk": "Unlicensed external assets enter the archive",
            "evidence": "No LICENSE file was found in the pinned CE2EPlan repository.",
            "response": (
                "Exclude CE2EPlan code, RPLAN data, and weights from package v12; "
                "include hashes, URLs, and run reports only."
            ),
            "residual": "Downstream users must obtain assets from the original releases.",
        },
        {
            "risk": "Locality is achieved by suppressing edits",
            "evidence": (
                "Outside exact remains 1.0 while edit-region response, Delta-IoU, "
                "Delta-MSE, Node F1, and edge overlap are reported separately."
            ),
            "response": "Always pair strict locality with inside-edit response and quality.",
            "residual": "Inside response varies by operation.",
        },
        {
            "risk": "The typed graph is overclaimed as trained attention",
            "evidence": (
                "The mechanism audit applies matched post-hoc routing arms to "
                "rendered outputs."
            ),
            "response": (
                "Call it typed routing/composition evidence, not a retrained "
                "cross-attention module."
            ),
            "residual": "A jointly trained typed mechanism remains future work.",
        },
        {
            "risk": "The control gate and structural graph are conflated",
            "evidence": (
                "No-gating violates outside exactness; graphless routing can "
                "preserve locality but loses edit and graph quality."
            ),
            "response": (
                "Attribute locality to control gating and quality to structural "
                "routing in separate claims."
            ),
            "residual": "Routing latency and watertight trade-offs remain.",
        },
        {
            "risk": "Watertightness is uniformly high",
            "evidence": (
                "Low-room-count, opening-only, and weak-interface bins can fall "
                "below 0.95 conditional watertightness."
            ),
            "response": "Define these bins as the current protocol boundary.",
            "residual": "Weak edit signals remain difficult.",
        },
        {
            "risk": "Oracle controls are confused with deployment",
            "evidence": "The structural main evidence uses oracle controls.",
            "response": (
                "Label oracle-control scope and keep missing/predicted-control "
                "results outside the primary claim."
            ),
            "residual": "Target-free structural deployment is incomplete.",
        },
        {
            "risk": "Five seeds are treated as population inference",
            "evidence": "Five deterministic seeds mainly support stability.",
            "response": "Do not infer population rank or unseen-data power from five seeds.",
            "residual": "Larger edit-set sweeps remain useful.",
        },
        {
            "risk": "Novelty is claimed as first-method priority",
            "evidence": "The novelty search does not exhaust all related systems.",
            "response": (
                "Use FEVP v1.0 and evaluated-matrix completeness, not a global "
                "first-method claim."
            ),
            "residual": "Broader protocol adoption and novelty coverage remain open.",
        },
        {
            "risk": "Human evaluation is absent",
            "evidence": "Human evaluation is excluded by project scope.",
            "response": (
                "Make no participant, preference, or usability claim; rely only "
                "on automated auditable evidence."
            ),
            "residual": "Subjective design quality is outside this evidence package.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v4",
        "",
        "Final FEVP and external-audit risk matrix. This is not a manuscript.",
        "",
        "| Risk | Evidence | Response | Residual risk |",
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
            "## Safe Claim",
            "",
            f"> {evidence['primary_narrative']}",
            "",
            "## Coverage Status",
            "",
            (
                f"- FloorFlow complete row: "
                f"`{coverage['claims']['floorflow_complete_under_fevp']}`."
            ),
            (
                f"- Matched Raster2Seq editing rows: "
                f"`{coverage['claims']['matched_floorflow_and_raster2seq']}`."
            ),
            (
                f"- Raster2Seq official 503 identity match: "
                f"`{coverage['claims']['raster2seq_503_matched_identity']}`."
            ),
            (
                f"- CE2EPlan same-data match: "
                f"`{coverage['claims']['ce2eplan_same_data_comparison']}`."
            ),
            "- Human evaluation: excluded, not pending.",
            "",
            "## Evidence Pointers",
            "",
            "- Protocol: `docs/fevp_v1_0.md`.",
            "- Matrix: `outputs/reports/fevp_coverage_matrix_v1.json`.",
            "- Mechanism: `outputs/reports/fp_locality_mechanism_v2.json`.",
            "- Raster2Seq: `outputs/reports/fp_raster2seq_official_anchor_v1.json`.",
            "- CE2EPlan: `outputs/reports/ce2eplan_official_anchor_v1.json`.",
            "",
        ]
    )
    _ = mechanism
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
