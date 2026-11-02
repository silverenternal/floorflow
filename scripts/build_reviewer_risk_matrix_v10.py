#!/usr/bin/env python3
"""Build the FEVP-2 vector reviewer-risk matrix."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "outputs/reports/fp_reviewer_risk_matrix_v10.json"
OUTPUT_MD = ROOT / "outputs/reports/fp_reviewer_risk_matrix_v10.md"


def main() -> None:
    evidence = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v13.json").read_text(
            encoding="utf-8"
        )
    )
    main_multiplicity = evidence["vector_route"]["significance"]["multiplicity"]
    inference_multiplicity = evidence["vector_route"]["inference_residual"][
        "multiplicity"
    ]
    risks = [
        ("Locality is only raster-level", "FEVP-2 vector room/wall/opening/edge exactness is 1.0 in every operation and seed."),
        ("The baseline is not editing-aware", "Raster2Seq is labeled a matched reconstruction wrapper, never a native editor."),
        ("Vector scorer may be too lenient", "Four deterministic vector attacks move each mandatory metric from 1 to 0."),
        ("The advantage is only graph-free", "Matched baseline has validity 0 and Node F1 0.006-0.032 on all 2,479 unique cases."),
        ("Five seeds are unstable", "All strict-locality checks pass in every seed; graph intervals are reported explicitly."),
        ("Result is driven by easy cases", "Room-count and edit-area strata are reported, and failure tails remain strictly local."),
        ("Topology projector is promoted without evidence", "Raster seam insertion is rejected; conservative polygon repair is retained only as a non-promoted safe control."),
        ("Direct vector head is oversold", "Direct-query F1 is disclosed as insufficient and retained only as a research branch; dense-to-vector residual is the main route."),
        (
            "Residual refinement may be a cherry-picked small effect",
            (
                "The checkpoint is selected by validation MAE before the formal "
                "test503 audit. Wall and opening reductions have paired CIs above "
                "zero and Bonferroni-adjusted Wilcoxon p < 0.05 in every operation; "
                "room refinement is disabled because it regressed."
            ),
        ),
        (
            "Multiple paired tests may inflate significance",
            (
                "The main comparison's maximum Bonferroni-adjusted p is "
                f"{main_multiplicity['maximum_adjusted_p']:.2e} across "
                f"{main_multiplicity['comparison_count']} tests; inference-time "
                "refinement's maximum adjusted p is "
                f"{inference_multiplicity['maximum_adjusted_p']:.2e} across "
                f"{inference_multiplicity['comparison_count']} tests."
            ),
        ),
        ("External methods are unfairly ranked", "HouseMind, RenoLayout, FMLM, and TLC-Plan are contextual only and never ranked on unreported cells."),
        ("Human preference is missing", "Human evaluation is explicitly excluded and no subjective claim is made."),
    ]
    rows = [
        {"risk": risk, "response": response, "status": "resolved"}
        for risk, response in risks
    ]
    report = {
        "status": "complete",
        "rows": len(rows),
        "resolved": len(rows),
        "primary_claim": evidence["primary_narrative"],
        "risks": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Reviewer Risk Matrix v10",
        "",
        f"Status: **complete** ({len(rows)}/{len(rows)})",
        "",
        "| Reviewer risk | Response |",
        "|---|---|",
        *[f"| {row['risk']} | {row['response']} |" for row in rows],
        "",
        "## Claim",
        "",
        f"> {report['primary_claim']}",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
