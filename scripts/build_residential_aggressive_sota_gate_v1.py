"""Summarize the five aggressive CVPR targets and their evidence status."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "identity": "outputs/reports/residential_identity_test509_v4_summary.json",
    "house_mind": "outputs/reports/housemind_inference_feasibility_v1.json",
    "free_text": "outputs/reports/housemind_free_text_editor_holdout_v1.json",
    "graph": "outputs/reports/residential_graph_protocol_dominance_v1.json",
    "rplan": "outputs/reports/rplan_full_cad_sota_gate_v1.json",
}


def main() -> None:
    payloads = {
        name: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for name, path in REPORTS.items()
    }
    free_text = payloads["free_text"]["summary"]
    identity = payloads["identity"]
    outcomes = {
        "beat_house_mind_e_complete_method": {
            "status": "blocked",
            "reason": payloads["house_mind"]["decision"],
        },
        "rplan_sota": {
            "status": "blocked",
            "reason": payloads["rplan"]["decision"],
        },
        "floorset_sota": {
            "status": "blocked",
            "reason": (
                "FloorSet is chip-floorplan work and is excluded by the "
                "user's no-chips instruction."
            ),
        },
        "complete_cad_sota": {
            "status": "blocked",
            "reason": (
                "No matched RPLAN/full-CAD data or executable baseline; "
                "current target-free CAD diagnostics are substantially weaker."
            ),
        },
        "free_text_editing_sota": {
            "status": "failed",
            "reason": (
                "Held-out deterministic free-text editor reaches "
                f"Delta-IoU {free_text['delta_iou']['mean']:.6f} and "
                f"Delta-MSE {free_text['delta_mse']['mean']:.6f}, below "
                "the published HouseMind-E anchors."
            ),
        },
        "all_graph_protocol_sota": {
            "status": "blocked",
            "reason": payloads["graph"]["decision"],
        },
        "paired_room_iou_improvement": {
            "status": "passed",
            "reason": identity["claim"],
            "paired_ci95": [
                identity["paired_test_vs_count_consensus"]["room_iou"][
                    "ci_low"
                ],
                identity["paired_test_vs_count_consensus"]["room_iou"][
                    "ci_high"
                ],
            ],
            "wilcoxon_p_greater": identity[
                "paired_test_vs_count_consensus"
            ]["room_iou"]["wilcoxon_p_greater"],
        },
    }
    report = {
        "status": "partial_supported_with_blockers",
        "scope": "residential_floorplan_only aggressive CVPR target audit",
        "outcomes": outcomes,
        "checks": {
            "room_iou_target_passed": (
                outcomes["paired_room_iou_improvement"]["status"] == "passed"
            ),
            "house_mind_defeat_not_claimed": (
                outcomes["beat_house_mind_e_complete_method"]["status"]
                == "blocked"
            ),
            "rplan_or_cad_sota_not_claimed": (
                outcomes["rplan_sota"]["status"] == "blocked"
                and outcomes["complete_cad_sota"]["status"] == "blocked"
            ),
            "free_text_sota_not_claimed": (
                outcomes["free_text_editing_sota"]["status"] == "failed"
            ),
            "all_protocol_sota_not_claimed": (
                outcomes["all_graph_protocol_sota"]["status"] == "blocked"
            ),
            "floorset_untouched": (
                outcomes["floorset_sota"]["status"] == "blocked"
            ),
        },
        "decision": (
            "Only the paired Room-IoU target passes. The other requested "
            "leaderboard and complete-method claims are unsupported and must "
            "not be written as SOTA."
        ),
        "sources": REPORTS,
    }
    output = ROOT / "outputs/reports/residential_aggressive_sota_gate_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Aggressive CVPR Target Audit",
        "",
        "| Requested target | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, values in outcomes.items():
        lines.append(
            f"| {name} | {values['status']} | {values['reason']} |"
        )
    table = ROOT / "paper/tables/residential_aggressive_sota_gate_v1.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
