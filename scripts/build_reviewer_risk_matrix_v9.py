#!/usr/bin/env python3
"""Build the final reviewer-risk matrix from ledger v12."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "outputs/reports/fp_reviewer_risk_matrix_v9.json"
OUTPUT_MD = ROOT / "outputs/reports/fp_reviewer_risk_matrix_v9.md"

LABELS = {
    "1.1": "CI method is ambiguous",
    "1.2": "Paired tests lack effect sizes",
    "1.3": "Prompt evidence lacks category breakdown",
    "1.4": "Seed dispersion lacks ranges",
    "1.5": "Significance language lacks p-values or CI",
    "2.1": "Typed/no-graph contrast is compute-confounded",
    "2.2": "No-gate leakage may be a seed accident",
    "2.3": "Room-deletion GED tail lacks uncertainty",
    "3.1": "Raster2Seq preprocessing is overstated",
    "3.2": "Outside-frozen wrapper is not reproducible",
    "3.3": "CE2EPlan sample selection is undisclosed",
    "3.4": "HouseMind blocked status is undated",
    "4.1": "FEVP definitions are informal",
    "4.2": "Outside exactness tolerance is unclear",
    "4.3": "Validity is an opaque aggregate",
    "4.4": "Attack suite has metric blind spots",
    "5.1": "Environment lockfiles are incomplete",
    "5.2": "Artifacts lack generator provenance",
    "5.3": "CE2EPlan adapter may use private APIs",
    "5.4": "Archive is not deterministic",
    "6.1": "Canonical text implies a HouseMind defeat",
    "6.2": "Generation and editing evidence are conflated",
    "6.3": "Protocol and model contributions are conflated",
    "6.4": "Related work positioning is unfair",
}

RESPONSES = {
    "1.1": "Student-t seed intervals and paired-bootstrap metadata are explicit.",
    "1.2": "Rank-biserial correlation and Cliff's delta accompany every Wilcoxon row.",
    "1.3": "Four predeclared prompt proxies each contain exactly 100 rows.",
    "1.4": "Every five-seed metric includes min/max seed range.",
    "1.5": "Canonical text audit requires p or CI beside significance language.",
    "2.1": "A same-checkpoint, zero-parameter, same-NFE budget control is reported.",
    "2.2": "No-gate leakage occurs in all five seeds with explicit seed values.",
    "2.3": "The 2,500-case deletion tail has a Student-t GED interval.",
    "3.1": "Official and shared-project preprocessing are separate fields.",
    "3.2": "The wrapper path and SHA-256 are published and audited.",
    "3.3": "Evenly spaced indices, method, and index digest are released.",
    "3.4": "A verification timestamp and four public-page hashes are recorded.",
    "4.1": "The appendix gives equations and pseudocode for every metric family.",
    "4.2": "Pixel tolerance is exactly zero and a one-pixel attack is reported.",
    "4.3": "Validity is the conjunction of five named checks with error counts.",
    "4.4": "Supplemental attacks cover Delta-IoU, Delta-MSE, and hole area.",
    "5.1": "pyproject, uv.lock, Python version, and external requirements are included.",
    "5.2": "Every canonical JSON/table maps to generator and input hashes.",
    "5.3": "The adapter imports public JSON files and no network/private API.",
    "5.4": "Sorted content, fixed metadata, and mtime-zero gzip are hash-verified.",
    "6.1": "Canonical text explicitly declines a HouseMind defeat claim.",
    "6.2": "CE2EPlan is labeled generation-only and outside exactness is N/A.",
    "6.3": "FEVP protocol and FloorFlow model claims have separate source fields.",
    "6.4": "HouseMind tokenization/editing and CE2EPlan topology are cited fairly.",
}


def main() -> None:
    ledger = json.loads(
        (ROOT / "outputs/reports/fp_cvpr_evidence_v12.json").read_text(
            encoding="utf-8"
        )
    )
    closure = json.loads(
        (ROOT / "outputs/reports/cvpr_risk_closure_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        {
            "id": risk_id,
            "risk": LABELS[risk_id],
            "resolved": closure["checks"][risk_id],
            "response": RESPONSES[risk_id],
            "evidence": closure["evidence"][risk_id],
        }
        for risk_id in LABELS
    ]
    report = {
        "status": "complete" if all(row["resolved"] for row in rows) else "failed",
        "rows": len(rows),
        "resolved": sum(row["resolved"] for row in rows),
        "submission_status": ledger["submission_status"],
        "primary_claim": ledger["primary_narrative"],
        "risks": rows,
        "boundaries": [
            "HouseMind is unreported under FEVP, not failed.",
            "CE2EPlan is a generation-only portability target.",
            "Raster2Seq matched rows use shared project preprocessing.",
            "Watertight guard is not a net-benefit contribution.",
            "Human preference is outside scope.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Reviewer Risk Matrix v9",
        "",
        f"Status: **{report['status']}** ({report['resolved']}/{report['rows']} resolved)",
        "",
        "| ID | Reviewer risk | Resolution | Evidence |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['id']} | {row['risk']} | {row['response']} | "
        f"`{row['evidence']}` |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Claim",
            "",
            f"> {report['primary_claim']}",
            "",
            "## Boundaries",
            "",
            *[f"- {boundary}" for boundary in report["boundaries"]],
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "resolved": report["resolved"]}))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
