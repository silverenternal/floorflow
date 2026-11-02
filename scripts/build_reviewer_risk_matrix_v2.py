#!/usr/bin/env python3
"""Build the consolidated reviewer-risk matrix from v5 and chip v3 ledgers."""

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
        default=ROOT / "outputs/reports/fp_reviewer_risk_matrix_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    floorplan = read("outputs/reports/fp_cvpr_evidence_v5.json")
    chip = read("outputs/reports/chip_cvpr_evidence_v3.json")
    claims = floorplan["primary_claim_evidence"]
    edit = claims["edit_metrics"]["aggregate"]["local_delta_iou"]
    mse = claims["edit_metrics"]["aggregate"]["local_delta_mse"]
    verifiability = claims["verifiability"]["floorflow"]["summary"]["verifiability"]
    structural = claims["housemind_style_structure"]
    after = [
        seed["summary"]["after_prediction_vs_after_target"]
        for seed in structural["per_seed"].values()
    ]
    node_f1 = [row["node_f1"] for row in after]
    edge_overlap = [row["edge_overlap"] for row in after]
    ged = [row["ged"] for row in after]
    factorial = claims["safety_factorial"]
    boundary = claims["protocol_boundary"]
    gap = chip["trinity_gap_decomposition"]
    gap_values = gap["headline_audit"]["predeclared_gap"]
    triage = chip["metric_triage"]
    rows = [
        {
            "risk": "HouseMind-E uses a stronger natural-language interface",
            "evidence": (
                "HouseMind-E RPLAN Table 3 is retained as a protocol anchor; "
                "FloorFlow uses control-gated CubiCasa rows."
            ),
            "response": (
                "Do not claim a same-benchmark win or complete natural-language "
                "method superiority."
            ),
            "residual": "HouseMind official inference code remains unreleased.",
        },
        {
            "risk": "Structural numbers are self-defined",
            "evidence": (
                f"HouseMind-style after-edit Node F1 {min(node_f1):.4f}-"
                f"{max(node_f1):.4f}, edge overlap {min(edge_overlap):.4f}-"
                f"{max(edge_overlap):.4f}, GED {min(ged):.4f}-{max(ged):.4f}; "
                "source-mapped in the v5 ledger."
            ),
            "response": (
                "Label the result protocol-aligned oracle-control evidence, "
                "not an RPLAN leaderboard rank."
            ),
            "residual": "No external method was run under the exact graph protocol.",
        },
        {
            "risk": "Oracle controls inflate structural results",
            "evidence": (
                "The main structural table explicitly uses oracle controls and "
                "a HouseMind-style room-type graph."
            ),
            "response": (
                "Keep predicted-control and target-free results outside the "
                "strict-locality main claim."
            ),
            "residual": "Target-free control deployment remains incomplete.",
        },
        {
            "risk": "Strict locality is achieved by suppressing editing",
            "evidence": (
                "Control-on/guard-off keeps outside pixel exactness 1.0 while "
                "reports a measurable inside-change response; the raw renderer "
                "does not preserve outside exactness."
            ),
            "response": (
                "Report locality and inside response together; do not infer "
                "quality from exactness alone."
            ),
            "residual": "Inside response varies by edit operation and seed.",
        },
        {
            "risk": "Outside exactness is only a post-hoc wrapper",
            "evidence": (
                "The editing-aware Raster2Seq outside-frozen wrapper remains "
                "far below FloorFlow on Delta-IoU and Delta-MSE."
            ),
            "response": (
                "Include the wrapper as a weak editing-aware baseline; label "
                "native Raster2Seq as reconstruction-only."
            ),
            "residual": "No native editing-aware external method was matched.",
        },
        {
            "risk": "Architecture ablations confuse locality and topology",
            "evidence": (
                "The 2x2 audit makes control gate the locality mechanism. "
                "The guard effect on watertightness is "
                f"{factorial['effects']['guard_on_minus_off']['conditional_watertight']['mean_delta']:+.6f} "
                "with a CI crossing zero."
            ),
            "response": (
                "Claim audited control-gate locality; do not claim a stable "
                "aggregate watertight-guard improvement."
            ),
            "residual": "The guard may interact with a subset of cases.",
        },
        {
            "risk": "Weak edit signals hide failures",
            "evidence": (
                f"Low-room-count watertightness is "
                f"{boundary['weak_boundaries']['low_room_count_0_4']['metrics']['conditional_watertight']:.4f}; "
                f"weak-interface watertightness is "
                f"{boundary['weak_boundaries']['weak_interface_opening_or_none']['metrics']['conditional_watertight']:.4f}; "
                "outside pixel and room-label exactness remain 1.0."
            ),
            "response": (
                "Define these bins as the current protocol boundary rather "
                "than hiding them in an average."
            ),
            "residual": "Sparse edit signals remain a deployment limitation.",
        },
        {
            "risk": "Five seeds are insufficient for population inference",
            "evidence": (
                f"Five seeds cover {claims['edit_metrics']['seeds']}; the "
                "claim uses them as stability evidence with row-level CIs."
            ),
            "response": (
                "Do not present five seeds as independent population sampling."
            ),
            "residual": "A larger edit-set sweep remains useful.",
        },
        {
            "risk": "The external comparator is not an editing baseline",
            "evidence": (
                "Raster2Seq is a matched CubiCasa reconstruction checkpoint; "
                "the outside-frozen wrapper is separately labeled editing-aware."
            ),
            "response": (
                "State reconstruction-baseline provenance and avoid claiming "
                "a native editing-method benchmark win."
            ),
            "residual": "A matched native editing method is still absent.",
        },
        {
            "risk": "Verifiability is mistaken for performance",
            "evidence": (
                f"Delta-IoU is {edit['min']:.4f}-{edit['max']:.4f} and "
                f"Delta-MSE is {mse['min']:.6f}-{mse['max']:.6f}; "
                f"Verifiability ranges {verifiability['min']:.4f}-"
                f"{verifiability['max']:.4f}; it multiplies four measured "
                "safety factors and returns N/A for incomplete external rows."
            ),
            "response": (
                "Always report the four factors before the composite; never "
                "use Verifiability to replace Delta-IoU/Delta-MSE."
            ),
            "residual": "Composite scores depend on the declared protocol.",
        },
        {
            "risk": "Chip evidence overclaims SOTA",
            "evidence": (
                f"Predeclared Trinity gaps remain {100 * gap_values['cost']:.2f}% "
                f"hard cost and {100 * gap_values['v_rel']:.2f}% V_rel."
            ),
            "response": (
                "Claim matched mechanism, tail concentration, and budget "
                "diagnostics only; retain Trinity as a stronger comparator."
            ),
            "residual": "The remaining gap is structural/geometric and partly unresolved.",
        },
        {
            "risk": "Weak chip metrics are promoted after the fact",
            "evidence": (
                "MIB has no official-panel signal; boundary and p2b intervals "
                "cross zero; high-b2b/high-pin strata are limitations."
            ),
            "response": (
                "Use the v1 metric triage as a claim gate and preserve no-claim "
                "status."
            ),
            "residual": "A future stronger benchmark may change the panel.",
        },
        {
            "risk": "Novelty is claimed without exhaustive venue coverage",
            "evidence": (
                "The claim is framed as an unusually complete auditable "
                "combination, not a first-method priority claim."
            ),
            "response": (
                "Keep the novelty boundary downgraded and cite the verified "
                "related-work sources."
            ),
            "residual": "Full venue-level novelty coverage remains incomplete.",
        },
    ]
    lines = [
        "# Reviewer Risk Matrix v2",
        "",
        (
            "Scope: consolidated experiment evidence. This file is reviewer "
            "guidance, not a manuscript."
        ),
        "",
        "| Risk | Evidence available | Claim response | Residual risk |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['risk']} | {row['evidence']} | "
            f"{row['response']} | {row['residual']} |"
        )
    lines.extend(
        [
            "",
            "## Safe Main-Claim Sentence",
            "",
            f"> {floorplan['main_claim']}",
            "",
            (
                "The contribution is an auditable strict-locality protocol and "
                "matched evidence combination. It is not a claim of HouseMind-E "
                "complete-method superiority or complete CAD SOTA."
            ),
            "",
            "## Chip Boundary Sentence",
            "",
            f"> {chip['allowed_boundary']}",
            "",
            "## Mandatory Limitation Sentence",
            "",
            f"> {floorplan['required_boundary_sentence']}",
            "",
            "## Claim-Ledger Checks",
            "",
            f"- Floorplan status: `{floorplan['status']}`.",
            f"- Chip status: `{chip['status']}`.",
            (
                "- Floorplan v5 promotion-gate failures: "
                f"`{[name for name, value in floorplan['new_gate_checks'].items() if value is False]}`; "
                "legacy non-claims remain recorded as disclosed limitations."
            ),
            f"- Chip metric triage checks: `{triage['checks']}`.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "risks": len(rows),
                "output": str(args.output.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
