"""Re-review residential external baselines and CVPR claim blockers."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "identity": "outputs/reports/residential_identity_test509_v4_summary.json",
    "house_mind_partial": "outputs/reports/housemind_partial_table_v1.json",
    "house_mind_feasibility": (
        "outputs/reports/housemind_inference_feasibility_v1.json"
    ),
    "rplan": "outputs/reports/rplan_full_cad_sota_gate_v1.json",
    "free_text": "outputs/reports/housemind_free_text_editor_holdout_v1.json",
    "graph": "outputs/reports/residential_graph_protocol_dominance_v1.json",
    "ce2eplan": "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
    "raster2seq_sensitivity": "outputs/reports/raster2seq_sensitivity_v1.json",
}


def main() -> None:
    payloads = {
        name: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for name, path in SOURCES.items()
    }
    free_text = payloads["free_text"]["summary"]
    identity = payloads["identity"]["paired_test_vs_count_consensus"][
        "room_iou"
    ]
    report = {
        "status": "rereview_complete_with_blockers",
        "scope": "residential_floorplan_only",
        "blockers": {
            "house_mind_complete_method": {
                "status": "blocked_after_partial_reproduction",
                "asset_hashes_verified": True,
                "model_weight_probe": True,
                "token_generation_probe": True,
                "full_image_inference": False,
                "reason": payloads["house_mind_feasibility"][
                    "missing_contract"
                ],
            },
            "rplan_sota": {
                "status": "blocked",
                "reason": payloads["rplan"]["decision"],
            },
            "floorset_sota": {
                "status": "excluded",
                "reason": (
                    "The user assigned chips work to a separate AI and "
                    "forbade touching chips code/results in this thread."
                ),
            },
            "free_text_editing_sota": {
                "status": "failed",
                "delta_iou": free_text["delta_iou"]["mean"],
                "delta_mse": free_text["delta_mse"]["mean"],
                "reason": (
                    "The deterministic free-text baseline remains below the "
                    "published HouseMind-E anchors."
                ),
            },
            "all_graph_protocol_sota": {
                "status": "blocked",
                "reason": payloads["graph"]["decision"],
            },
        },
        "new_evidence": {
            "house_mind_partial_reproduction": SOURCES[
                "house_mind_partial"
            ],
            "raster2seq_matched_sensitivity": SOURCES[
                "raster2seq_sensitivity"
            ],
            "ce2eplan_tvcg_anchor": SOURCES["ce2eplan"],
            "paired_room_iou": {
                "mean_delta": identity["mean_delta"],
                "ci95": [identity["ci_low"], identity["ci_high"]],
                "wilcoxon_p_greater": identity["wilcoxon_p_greater"],
            },
        },
        "allowed_claims": [
            "Raster2Seq sensitivity analysis under one matched preprocessing contract.",
            "Room-IoU improvement with paired test statistics.",
            "CE2EPlan as a generation-only RPLAN reference anchor.",
            "HouseMind published scores as reference-ceiling context only.",
        ],
        "forbidden_claims": [
            "FloorFlow beats HouseMind-E's complete method.",
            "FloorFlow is matched to HouseMind's RPLAN editing benchmark.",
            "FloorFlow is RPLAN, FloorSet, or complete-CAD SOTA.",
            "CE2EPlan and FloorFlow share a leaderboard.",
            "Graph protocols can be pooled into one unified SOTA score.",
        ],
        "decision": (
            "Keep Room IoU and matched Raster2Seq sensitivity as the strongest "
            "defensible additions. Preserve all external leaderboard claims as "
            "blocked or context-only."
        ),
    }
    output = ROOT / "outputs/reports/residential_blocker_rereview_v2.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Residential Blocker Re-Review",
        "",
        "| Item | Status |",
        "|---|---|",
    ]
    for name, values in report["blockers"].items():
        lines.append(f"| {name} | {values['status']} |")
    lines.extend(
        [
            "",
            (
                "HouseMind has a partial reproduction only: public weights load "
                "and emit spatial tokens, but the official image-to-token "
                "through post-processing chain remains unavailable."
            ),
        ]
    )
    table = ROOT / "paper/tables/residential_blocker_rereview_v2.md"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
