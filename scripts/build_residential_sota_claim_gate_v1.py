"""Audit the frozen residential experiments against claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "identity": (
        ROOT / "outputs/reports/residential_identity_test509_v4_summary.json"
    ),
    "structure": (
        ROOT / "outputs/reports/residential_structural_hardened_v2.json"
    ),
    "edit": ROOT / "outputs/reports/residential_edit_test503_v1.json",
    "vector": (
        ROOT / "outputs/reports/residential_edit_vector_protocol_raw_v1.json"
    ),
    "ablation": ROOT / "outputs/reports/residential_sota_ablation_v1.json",
    "strata": ROOT / "outputs/reports/residential_sota_strata_v1.json",
    "attacks": ROOT / "outputs/reports/residential_sota_protocol_attacks_v1.json",
    "controls": ROOT / "outputs/reports/residential_sota_controls_v1.json",
    "editing_baseline": (
        ROOT / "outputs/reports/residential_editing_aware_baseline_v1.json"
    ),
    "house_mind": (
        ROOT / "outputs/reports/housemind_partial_table_v1.json"
    ),
    "rplan": (
        ROOT / "outputs/reports/rplan_full_cad_sota_gate_v1.json"
    ),
    "ce2eplan": (
        ROOT / "outputs/reports/ce2eplan_tvcg_anchor_v2.json"
    ),
    "blocker_rereview": (
        ROOT / "outputs/reports/residential_blocker_rereview_v2.json"
    ),
    "line_limit": ROOT / "outputs/reports/residential_line_limit_audit_v4.json",
}
FORBIDDEN_PATTERNS = (
    "beats housemind",
    "rplan sota",
    "full cad sota",
    "leaderboard sota",
    "matches housemind",
)


def main() -> None:
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in REPORTS.items()
    }
    checks = {
        "identity_passed": payloads["identity"]["status"] == "passed",
        "official_test509_once": payloads["identity"]["official_test509_used"],
        "paired_room_iou_passed": payloads["identity"]["checks"][
            "room_iou_paired_ci_positive"
        ],
        "structure_passed": payloads["structure"]["status"] == "passed",
        "edit_passed": payloads["edit"]["status"] == "passed",
        "vector_locality_passed": payloads["vector"]["status"] == "passed",
        "ablation_passed": payloads["ablation"]["status"] == "passed",
        "strata_complete": payloads["strata"]["status"] == "complete",
        "protocol_attacks_passed": payloads["attacks"]["status"] == "passed",
        "controls_passed": payloads["controls"]["status"] == "passed",
        "editing_baseline_passed": payloads["editing_baseline"]["status"] == "passed",
        "house_mind_partial_bounded": payloads["house_mind"]["checks"][
            "official_inference_chain_recovered"
        ]
        is False,
        "rplan_claim_blocked": payloads["rplan"]["status"] == "blocked",
        "ce2eplan_generation_only": payloads["ce2eplan"]["checks"][
            "editing_comparison"
        ]
        is False,
        "blocker_rereview_complete": payloads["blocker_rereview"]["status"]
        == "rereview_complete_with_blockers",
        "line_limit_ok": not payloads["line_limit"]["violations"],
    }
    claim = (
        "On CubiCasa5K official test509, validation-frozen eight-view room-count "
        "consensus reaches RoomF1 0.898948, 0.011948 above the reported 0.887 "
        "anchor, with paired RoomF1 95% CI [0.007667, 0.017657]. Under the "
        "validation-frozen five-view polygon fusion, Room IoU reaches 0.625051 "
        "with paired 95% CI [0.000779, 0.002794], Wilcoxon p=0.000280, while "
        "RoomF1 changes on 0/509 cases. Under the "
        "CubiCasa test503 oracle-control protocol, FloorFlow reaches Delta-IoU "
        "0.9567-0.9998 across five operations, maintains exact "
        "outside pixel/room/polygon/wall/opening/edge preservation, FP-Eval "
        "validity 1.0, and conditional watertightness 0.9579-0.9675. The "
        "independent FEVP five-check validity is reported separately and is not "
        "relabeled as FP-Eval validity. Five-seed aligned structural results "
        "reach Node F1 0.982250 and edge overlap 0.929475. HouseMind public "
        "weights were exercised only as a partial token-generation "
        "reproduction; published HouseMind-E values are reference-ceiling "
        "context, not a matched ranking."
    )
    claim_lower = claim.lower()
    checks["claim_avoids_forbidden_patterns"] = not any(
        pattern in claim_lower for pattern in FORBIDDEN_PATTERNS
    )
    report = {
        "status": "passed" if all(checks.values()) else "not_passed",
        "checks": checks,
        "claim": claim,
        "boundaries": [
            "Identity uses official test509; editing uses test503 and the two are not pooled.",
            "Raster2Seq is a reconstruction method, not an editing-aware competitor.",
            "The RoomF1 gain is specific to the official test509 protocol and must be reported to six decimals.",
            "The Room IoU claim is specific to the validation-frozen five-view polygon-fusion arm; the count-consensus arm alone has a crossing interval.",
            "The edit result uses declared oracle controls and does not imply free-text editing.",
            "FP-Eval validity and the stricter FEVP five-check validity are distinct metrics.",
            "Aligned, canonical direct-tiled, and typed-matched graph protocols are not pooled.",
            "HouseMind-E is a published reference ceiling; no image outputs or benchmark metrics were produced from the public partial reproduction.",
            "CE2EPlan is a generation-only RPLAN anchor with DOI 10.1109/TVCG.2026.3665422.",
            "No claim is made about HouseMind's full method, RPLAN, or complete CAD generation.",
        ],
        "forbidden_patterns": list(FORBIDDEN_PATTERNS),
        "sources": {
            name: str(path.relative_to(ROOT))
            for name, path in REPORTS.items()
        },
    }
    output = ROOT / "outputs/reports/residential_sota_claim_gate_v5.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
