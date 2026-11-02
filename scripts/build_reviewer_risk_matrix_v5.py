#!/usr/bin/env python3
"""Build the reviewer-risk matrix for the hardened FEVP revision."""

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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v5.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v8.json").read_text(
            encoding="utf-8"
        )
    )
    hardened = evidence["hardened_fevp"]
    validity = hardened["validity_watertight"]["summary"]["all"]
    outside = hardened["outside"]["summary"]["all"]
    graph = hardened["graph"]["summary"]["seed_mean"]["typed_edit_graph"]
    risks = [
        {
            "risk": "The evaluator is too loose",
            "evidence": (
                "All mandatory metrics have deterministic attacks with zero "
                "observed false-pass rate."
            ),
            "response": "Publish the attack suite, rows, tolerances, and reference implementation.",
            "residual": "Synthetic attacks cannot cover every real failure mode.",
        },
        {
            "risk": "Outside exact is a single binary decoration",
            "evidence": (
                "Pixel, room-instance, polygon, edge, and continuous violation "
                "layers are reported separately."
            ),
            "response": "Claim the three-layer exactness only with the violation map.",
            "residual": "The strict arm is designed to preserve outside by construction.",
        },
        {
            "risk": "The negative control only edits easy cases",
            "evidence": (
                f"{hardened['outside']['summary']['single_room_negative_control']['cases']} "
                "single-room controls all preserve unrelated rooms."
            ),
            "response": "Report the control count and per-operation breakdown.",
            "residual": "Single-room controls do not represent global redesign edits.",
        },
        {
            "risk": "Validity is still a single fpeval flag",
            "evidence": (
                f"Hardened validity is {validity['validity']['validity']:.6f}, "
                "the conjunction of five explicit checks."
            ),
            "response": (
                "Replace the old validity 1.0 claim with the five-check result "
                "and overlap/semantic error counts."
            ),
            "residual": "A 2x2-pixel overlap tolerance remains a protocol choice.",
        },
        {
            "risk": "Watertightness is only a rate",
            "evidence": (
                f"Hole count {validity['watertight']['hole_count']:.6f}, "
                f"dangling rate {validity['watertight']['dangling_endpoint_rate']:.6f}, "
                "and source-linked failure panels are reported."
            ),
            "response": "Keep hole and dangling diagnostics in the appendix.",
            "residual": "Hole localization depends on raster topology extraction.",
        },
        {
            "risk": "The watertight guard is claimed as beneficial",
            "evidence": (
                f"Guard repairs {validity['repair_count']} but regresses "
                f"{validity['regression_count']}."
            ),
            "response": "Do not promote the guard; classify it as a failed net-benefit mechanism.",
            "residual": "A selective repair policy may work in future.",
        },
        {
            "risk": "Node F1 hides precision/recall imbalance",
            "evidence": (
                f"Node precision {graph['node_precision']:.6f}, recall "
                f"{graph['node_recall']:.6f}, F1 {graph['node_f1']:.6f}."
            ),
            "response": "Always report the three values together.",
            "residual": "Semantic type sets still collapse repeated room instances.",
        },
        {
            "risk": "Edge overlap hides edge precision/recall imbalance",
            "evidence": (
                f"Typed edge precision {graph['edge_precision']:.6f}, recall "
                f"{graph['edge_recall']:.6f}, F1 {graph['edge_f1']:.6f}."
            ),
            "response": "Report canonical graph P/R/F1 for all routing arms.",
            "residual": "Edge semantics remain a two-pixel contact proxy.",
        },
        {
            "risk": "GED is compared across incompatible evaluators",
            "evidence": "All GED claims use one fixed evaluator and paired tests.",
            "response": "Report only paired differences and p-values; never absolute cross-paper ranks.",
            "residual": "Timeout/fallback cases remain non-exact.",
        },
        {
            "risk": "Delta means are driven by easy samples",
            "evidence": "Distributions, hard decile, operation, room-count, and edit-area strata are reported.",
            "response": "Keep the hard-tail table and distribution figure in the paper package.",
            "residual": "No wall-resize or multi-step samples exist.",
        },
        {
            "risk": "Locality is evaluated only as 0/1",
            "evidence": (
                f"No-gate violation rate is {outside['no_edit_gating']['locality_violation_rate']:.6f}, "
                "while strict is zero."
            ),
            "response": "Report both exactness and continuous violation rate.",
            "residual": "Violation rate is sensitive to the valid-outside denominator.",
        },
        {
            "risk": "Five seeds are treated as population inference",
            "evidence": "Mean, SD, 95% CI, best/worst seed are reported.",
            "response": "Describe them as deterministic seed stability.",
            "residual": "Larger samples remain useful.",
        },
        {
            "risk": "External baselines are still unmatched",
            "evidence": "Raster2Seq case units and CE2EPlan task/data differ.",
            "response": "Keep external anchors separate and do not merge leaderboards.",
            "residual": "No native matched editing-aware baseline is available.",
        },
        {
            "risk": "Human preference is missing",
            "evidence": "Human evaluation is excluded by project scope.",
            "response": "Make no participant or preference claim.",
            "residual": "Subjective quality remains outside the automated protocol.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v5",
        "",
        "Hardened FEVP experiment evidence only.",
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
            "## Safe Claim",
            "",
            f"> {evidence['primary_narrative']}",
            "",
            "## Mandatory Boundaries",
            "",
            "- The old validity 1.0 is not the five-check validity claim.",
            "- The watertight guard has negative net benefit on the audited rows.",
            "- Five seeds are stability samples, not population sampling.",
            "- GED is interpreted only through paired tests under one evaluator.",
            "- Human evaluation is excluded.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
