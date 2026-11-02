#!/usr/bin/env python3
"""Build the strict-locality reviewer-risk matrix v3."""

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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v3.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = read("outputs/reports/fp_cvpr_evidence_v6.json")
    house = evidence["housemind_public_status"]
    protocol = evidence["protocol_v2"]
    typed = evidence["typed_edit_graph_ablation"]
    strata = evidence["edit_type_strata_v2"]
    pareto = evidence["dual_pareto_v2"]
    typed_summary = typed["per_seed"]["19"]["summary"]
    typed_arm = typed_summary["typed_edit_graph"]
    graphless = typed_summary["no_structural_graph"]
    risks = [
        {
            "risk": "HouseMind is treated as a like-for-like baseline",
            "evidence": "Official page marks code `Code (Soon)`; runnable demo is listed.",
            "response": "Describe HouseMind as complementary MLLM context, not a matched baseline.",
            "residual": "Official inference and token serialization remain unavailable.",
        },
        {
            "risk": "The protocol is mistaken for leaderboard SOTA",
            "evidence": "Main contribution is the strict-locality contract and auditable evidence.",
            "response": "Use protocol-level claims only; no HouseMind, FloorSet, or CAD SOTA.",
            "residual": "No native editing-aware external method is matched.",
        },
        {
            "risk": "Outside exactness is easy to obtain by hiding edits",
            "evidence": (
                f"Typed graph keeps outside exact at {typed_arm['outside_pixel_exact']:.3f} "
                f"while inside response is {typed_arm['inside_change_fraction']:.3f}."
            ),
            "response": "Always report outside exactness together with inside response and Delta-IoU.",
            "residual": "Inside response varies by operation.",
        },
        {
            "risk": "The typed graph is overclaimed as trained cross-attention",
            "evidence": "The four-arm experiment applies post-hoc routing to matched rendered outputs.",
            "response": "Call it typed routing/composition evidence, not a retrained attention mechanism.",
            "residual": "A jointly trained implementation remains future work.",
        },
        {
            "risk": "Typed graph harms watertightness",
            "evidence": (
                f"Typed {typed_arm['conditional_watertight']:.6f} vs graphless "
                f"{graphless['conditional_watertight']:.6f}; paired CI crosses zero."
            ),
            "response": "Report `supported_with_topology_tradeoff`; do not claim a watertight gain.",
            "residual": "The point estimate is slightly lower.",
        },
        {
            "risk": "Dense routing leaks outside the edit region",
            "evidence": (
                f"Dense outside exact {typed_summary['dense_cross_attention']['outside_pixel_exact']:.6f}; "
                "no-gating outside exact "
                f"{typed_summary['no_edit_gating']['outside_pixel_exact']:.6f}."
            ),
            "response": "Keep strict-locality and raw proposal leakage as separate measurements.",
            "residual": "The dense arm is a diagnostic, not a deployable method.",
        },
        {
            "risk": "GED range is unexplained",
            "evidence": (
                "Room removal explains the tail; edge insertion/deletion and node "
                "edits are reported separately and reconcile to GED."
            ),
            "response": "Report operation-level GED decomposition.",
            "residual": "No complete wall-move or resize graph analysis exists.",
        },
        {
            "risk": "Requested edit classes are missing",
            "evidence": "Wall move, resize, and sequential multi-step edits are unavailable.",
            "response": "Declare coverage gaps; make no numeric claim for missing classes.",
            "residual": "External edit coverage remains incomplete.",
        },
        {
            "risk": "Safety-efficiency favors the graphless arm",
            "evidence": (
                "The safety-efficiency frontier contains only no-structural-graph; "
                "typed routing is slower and has a lower watertight point estimate."
            ),
            "response": "Keep the typed graph claim on quality/locality/graph consistency, not efficiency.",
            "residual": "Retrained typed graph may change the cost trade-off.",
        },
        {
            "risk": "Weak bins are averaged away",
            "evidence": "Low-room-count and weak-interface bins fall below 0.95 conditional watertightness.",
            "response": "Keep the protocol boundary explicit and preserve outside exactness separately.",
            "residual": "Sparse edit signals remain difficult.",
        },
        {
            "risk": "Verifiability is mistaken for quality",
            "evidence": "Verifiability multiplies four safety factors and returns N/A when any factor is missing.",
            "response": "Report the four factors before the product; keep Delta-IoU separate.",
            "residual": "Composite values are protocol-dependent.",
        },
        {
            "risk": "Oracle controls are confused with deployment",
            "evidence": "Structural main results use oracle controls.",
            "response": "Label oracle-control scope and keep predicted controls outside the claim.",
            "residual": "Target-free structural deployment remains incomplete.",
        },
        {
            "risk": "Five seeds are treated as population inference",
            "evidence": "Five seeds are stability evidence with paired intervals.",
            "response": "Do not imply independent population sampling.",
            "residual": "Larger edit-set sweeps remain useful.",
        },
        {
            "risk": "Novelty is claimed as first-method priority",
            "evidence": "Novelty search coverage is explicitly incomplete.",
            "response": "Use audited-combination and protocol framing.",
            "residual": "Full venue-level novelty coverage remains open.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v3",
        "",
        "Consolidated strict-locality protocol evidence. This is not a manuscript.",
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
            "## Safe Protocol Claim",
            "",
            f"> {protocol_claim(protocol)}",
            "",
            "## Allowed HouseMind Wording",
            "",
            (
                "> HouseMind and FloorFlow target complementary aspects of "
                "floorplan generation: unified MLLM understanding/generation/"
                "editing versus a strict-locality editing protocol and "
                "verifiable evidence suite."
            ),
            "",
            "## Mandatory Limitations",
            "",
            f"> {evidence['required_boundary_sentence']}",
            "",
            "- The typed graph is post-hoc routing, not retrained attention.",
            "- Wall-move, resize, and multi-step edits are not numerically covered.",
            "- The safety-efficiency frontier favors the graphless arm.",
            "",
            "## Evidence Pointers",
            "",
            f"- Protocol: `{evidence['protocol_sources']['definition']}`.",
            f"- Typed graph: `{evidence['protocol_sources']['typed_graph']}`.",
            f"- Edit types: `{evidence['protocol_sources']['edit_types']}`.",
            f"- Dual Pareto: `{evidence['protocol_sources']['pareto']}`.",
            f"- HouseMind status: `{evidence['protocol_sources']['housemind']}`.",
            "",
        ]
    )
    _ = house, strata, pareto
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "risks": len(risks)}, indent=2))


def protocol_claim(protocol: dict) -> str:
    return (
        "FloorFlow turns edit locality into an auditable constraint: "
        "changes are allowed only inside the declared region, outside pixel "
        "and room-label exactness are measured, and validity, watertightness, "
        "editing quality, and graph consistency are reported separately "
        f"under `{protocol['version']}`."
    )


if __name__ == "__main__":
    main()
