#!/usr/bin/env python3
"""Build reviewer risks for the bounded offensive FEVP claim."""

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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v6.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v9.json").read_text(
            encoding="utf-8"
        )
    )
    final = evidence["offensive_evidence"]["final_claim"]
    locality = evidence["offensive_evidence"]["locality"]
    pareto = evidence["offensive_evidence"]["pareto"]
    safety = evidence["offensive_evidence"]["safety_baseline"]
    strength = evidence["offensive_evidence"]["edit_strength"]
    risks = [
        {
            "risk": "Locality 1.0 is dismissed as a constant-pass metric",
            "evidence": (
                f"No-gate room exactness is "
                f"{locality['no_edit_gating']['outside_room_level_exact']:.4f}; "
                f"violation rate is "
                f"{locality['no_edit_gating']['locality_violation_rate']:.4f}."
            ),
            "response": "Use leaking baselines as positive sensitivity evidence and show the heatmap.",
            "residual": "Strict preservation is partly architectural by construction.",
        },
        {
            "risk": "Single-room controls are too easy",
            "evidence": (
                f"{locality['single_room_negative_controls']:,} controls all "
                "preserve unrelated rooms."
            ),
            "response": "Report the count and operation breakdown.",
            "residual": "Global redesign edits remain outside this control.",
        },
        {
            "risk": "Typed routing is only an ablation",
            "evidence": (
                "Typed routing is the computed joint Pareto frontier over "
                "locality, Delta-IoU, Node F1, edge F1, and GED."
            ),
            "response": "Frame typed routing as a mechanism finding, not a cosmetic ablation.",
            "residual": "It remains a post-hoc matched routing analysis.",
        },
        {
            "risk": "Node F1 or edge F1 hides a precision-recall imbalance",
            "evidence": "Both metrics include precision, recall, and F1 in the graph report.",
            "response": "Report P/R/F1 together and avoid a single-score claim.",
            "residual": "Two-pixel edge contact is a protocol proxy.",
        },
        {
            "risk": "GED improvements are evaluator-dependent",
            "evidence": "All GED comparisons are paired under one fixed evaluator.",
            "response": "Report paired deltas and Wilcoxon tests, never absolute cross-paper ranks.",
            "residual": "Timeout/fallback rows remain nonexact.",
        },
        {
            "risk": "Validity is again overclaimed as 1.0",
            "evidence": (
                f"Hardened five-check validity is "
                f"{safety['validity']['validity']:.4f}."
            ),
            "response": "Keep validity as a safety baseline and report overlap/semantic failures.",
            "residual": "Grid-snap overlap tolerance is a protocol choice.",
        },
        {
            "risk": "Watertight guard is marketed as effective",
            "evidence": (
                f"{safety['repair_count']} repairs versus "
                f"{safety['regression_count']} regressions."
            ),
            "response": "Do not promote the guard; report localization diagnostics instead.",
            "residual": "A selective repair policy remains future work.",
        },
        {
            "risk": "Delta means hide difficult edit types",
            "evidence": (
                "1-room, 2-3-room, 4+-room, room-deletion, and opening/furniture "
                "groups are reported."
            ),
            "response": "Keep edit-strength strata and room-deletion GED tail in the appendix.",
            "residual": "Wall resize and multi-step edits remain unavailable.",
        },
        {
            "risk": "Protocol attacks are confused with model gains",
            "evidence": "The compact table states that attacks test evaluator sensitivity.",
            "response": "Keep the disclaimer beside the table.",
            "residual": "Synthetic attacks do not cover every real-world failure.",
        },
        {
            "risk": "External SOTA is claimed without matched evidence",
            "evidence": "HouseMind code is unavailable and external data/tasks differ.",
            "response": "Use only the evaluated or internally matched evidence described in the final claim.",
            "residual": "No native editing-aware external baseline is matched.",
        },
        {
            "risk": "Human preference is missing",
            "evidence": "Human evaluation is excluded by scope.",
            "response": "Make no subjective-quality or participant-derived claim.",
            "residual": "Subjective aesthetics remain unmeasured.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v6",
        "",
        "Bounded offensive FEVP evidence only.",
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
            "## Final Claim Boundary",
            "",
            f"> {final['claim_en']}",
            "",
            "## Forbidden",
            "",
            *[f"- {claim}" for claim in final["forbidden_claims"]],
            "",
            "## Evidence Pointers",
            "",
            "- Locality: `outputs/reports/fevp_locality_claim_v2.json`.",
            "- Pareto: `outputs/reports/fevp_mechanism_pareto_v2.json`.",
            "- Safety baseline: `outputs/reports/fevp_safety_baseline_v2.json`.",
            "- Edit strength: `outputs/reports/fevp_edit_strength_v2.json`.",
            "- Compact attacks: `outputs/reports/fevp_protocol_attacks_compact_v2.json`.",
            "",
        ]
    )
    _ = pareto, strength
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
