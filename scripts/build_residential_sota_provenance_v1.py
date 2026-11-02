"""Record software, data and artifact provenance for residential evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "configs/residential_sota_lockbox_v1.json",
    "configs/residential_identity_selected_v1.json",
    "configs/residential_identity_dihedral_consensus_v1.json",
    "configs/residential_room_iou_fusion_v1.json",
    "configs/fevp_metric_contract_v2.json",
    "outputs/reports/residential_identity_selection_val_v1.json",
    "outputs/reports/residential_identity_test509_v1.json",
    "outputs/reports/raster2seq_dihedral_consensus_val_v1.json",
    "outputs/reports/residential_identity_test509_v3_dihedral.json",
    "outputs/reports/residential_identity_test509_v3_summary.json",
    "outputs/reports/raster2seq_polygon_fusion_val_v5.json",
    "outputs/reports/residential_identity_test509_room_iou_v1.json",
    "outputs/reports/residential_identity_test509_v4_summary.json",
    "outputs/reports/residential_edit_test503_v1.json",
    "outputs/reports/residential_edit_vector_protocol_beforeonly_v1.json",
    "outputs/reports/residential_edit_vector_protocol_raw_v1.json",
    "outputs/reports/residential_editing_aware_baseline_v1.json",
    "outputs/reports/residential_structural_hardened_v2.json",
    "outputs/reports/residential_validity_reconciliation_v2.json",
    "outputs/reports/residential_sota_main_results_v4.json",
    "outputs/reports/residential_sota_ablation_v1.json",
    "outputs/reports/residential_sota_strata_v1.json",
    "outputs/reports/residential_sota_failures_v1.json",
    "outputs/reports/residential_sota_protocol_attacks_v1.json",
    "outputs/reports/residential_sota_controls_v1.json",
    "outputs/reports/residential_line_limit_audit_v4.json",
    "outputs/reports/residential_sota_claim_gate_v5.json",
    "outputs/reports/fevp_fiveseed_stability_v2.json",
    "outputs/reports/fevp_validity_watertight_v2.json",
    "outputs/reports/fevp_graph_metrics_v2.json",
    "outputs/reports/fp_control_gated_hard_gate_v1.json",
    "outputs/reports/housemind_released_asset_audit_v1.json",
    "outputs/reports/housemind_weight_probe_v1.json",
    "outputs/reports/housemind_inference_feasibility_v1.json",
    "outputs/reports/housemind_partial_reproduction_v1.json",
    "outputs/reports/housemind_partial_table_v1.json",
    "outputs/reports/housemind_free_text_editor_holdout_v1.json",
    "outputs/reports/residential_graph_protocol_dominance_v1.json",
    "outputs/reports/rplan_full_cad_sota_gate_v1.json",
    "outputs/reports/raster2seq_sensitivity_v1.json",
    "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
    "outputs/reports/residential_sota_main_results_v5.json",
    "outputs/reports/residential_blocker_rereview_v2.json",
    "outputs/reports/residential_aggressive_sota_gate_v1.json",
)
GENERATORS = {
    "outputs/reports/raster2seq_dihedral_consensus_val_v1.json": (
        "scripts/evaluate_raster2seq_dihedral_consensus_v1.py"
    ),
    "outputs/reports/residential_identity_test509_v3_dihedral.json": (
        "scripts/evaluate_raster2seq_dihedral_consensus_v1.py"
    ),
    "outputs/reports/residential_identity_test509_v3_summary.json": (
        "scripts/build_residential_identity_report_v3.py"
    ),
    "outputs/reports/residential_structural_hardened_v2.json": (
        "scripts/build_residential_structural_hardened_v2.py"
    ),
    "outputs/reports/residential_validity_reconciliation_v2.json": (
        "scripts/build_residential_structural_hardened_v2.py"
    ),
    "outputs/reports/residential_identity_test509_v4_summary.json": (
        "scripts/build_residential_identity_report_v4.py"
    ),
    "outputs/reports/residential_sota_main_results_v5.json": (
        "scripts/build_residential_sota_main_table_v2.py"
    ),
    "outputs/reports/residential_sota_claim_gate_v5.json": (
        "scripts/build_residential_sota_claim_gate_v1.py"
    ),
    "outputs/reports/residential_line_limit_audit_v4.json": (
        "scripts/audit_python_line_limit_v1.py"
    ),
    "outputs/reports/housemind_inference_feasibility_v1.json": (
        "scripts/build_housemind_inference_feasibility_v1.py"
    ),
    "outputs/reports/housemind_free_text_editor_holdout_v1.json": (
        "scripts/evaluate_housemind_free_text_editor_v1.py"
    ),
    "outputs/reports/residential_graph_protocol_dominance_v1.json": (
        "scripts/build_residential_graph_protocol_dominance_v1.py"
    ),
    "outputs/reports/rplan_full_cad_sota_gate_v1.json": (
        "scripts/build_rplan_full_cad_sota_gate_v1.py"
    ),
    "outputs/reports/residential_aggressive_sota_gate_v1.json": (
        "scripts/build_residential_aggressive_sota_gate_v1.py"
    ),
    "outputs/reports/housemind_partial_reproduction_v1.json": (
        "scripts/run_housemind_partial_reproduction_v1.py"
    ),
    "outputs/reports/housemind_partial_table_v1.json": (
        "scripts/build_housemind_partial_table_v1.py"
    ),
    "outputs/reports/raster2seq_sensitivity_v1.json": (
        "scripts/build_raster2seq_sensitivity_v1.py"
    ),
    "outputs/reports/ce2eplan_tvcg_anchor_v2.json": (
        "scripts/build_ce2eplan_tvcg_anchor_v2.py"
    ),
    "outputs/reports/residential_blocker_rereview_v2.json": (
        "scripts/build_residential_blocker_rereview_v2.py"
    ),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    artifacts = {
        path: {"sha256": sha256(ROOT / path), "bytes": (ROOT / path).stat().st_size}
        for path in ARTIFACTS
    }
    report = {
        "status": "complete",
        "scope": "residential_floorplan_only",
        "chips_artifacts_included": False,
        "environment": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "numpy": version("numpy"),
            "opencv": cv2.__version__,
            "shapely": version("shapely"),
            "scipy": version("scipy"),
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        },
        "repository": {
            "head": git_value("rev-parse", "HEAD"),
            "working_tree": (
                "repository has no baseline commit; package files are hashed directly"
            ),
        },
        "raster2seq": {
            "path": "third_party/Raster2Seq",
            "commit": git_value("-C", "third_party/Raster2Seq", "rev-parse", "HEAD"),
            "status": git_value("-C", "third_party/Raster2Seq", "status", "--short"),
        },
        "artifacts": artifacts,
        "generators": GENERATORS,
        "commands": {
            "identity_validation": (
                "PYTHONPATH=src uv run python "
                "scripts/evaluate_raster2seq_dihedral_consensus_v1.py"
            ),
            "identity_test_after_policy_freeze": (
                "PYTHONPATH=src uv run python "
                "scripts/evaluate_raster2seq_polygon_fusion_v1.py "
                "--eval-set test --run-prefix test509 "
                "--dataset-root .data/external/raster2seq/"
                "coco_cubicasa5k_nowalls_v4_test_refined "
                "--force-strategy support5_iou085_selected3"
            ),
            "identity_report": (
                "uv run python scripts/build_residential_identity_report_v4.py"
            ),
            "house_mind_weight_probe": (
                "CUDA_VISIBLE_DEVICES=0 uv run python "
                "scripts/probe_housemind_weights_v1.py"
            ),
            "free_text_holdout": (
                "PYTHONPATH=src uv run python "
                "scripts/evaluate_housemind_free_text_editor_v1.py "
                "--start-index 12"
            ),
            "house_mind_partial": (
                "CUDA_VISIBLE_DEVICES=0 uv run python "
                "scripts/run_housemind_partial_reproduction_v1.py"
            ),
            "raster2seq_sensitivity": (
                "uv run python scripts/build_raster2seq_sensitivity_v1.py"
            ),
            "ce2eplan_anchor": (
                "uv run python scripts/build_ce2eplan_tvcg_anchor_v2.py"
            ),
            "main_table": (
                "PYTHONPATH=src uv run python "
                "scripts/build_residential_sota_main_table_v2.py"
            ),
            "claim_gate": (
                "PYTHONPATH=src uv run python "
                "scripts/build_residential_sota_claim_gate_v1.py"
            ),
        },
    }
    output = ROOT / "outputs/reports/residential_sota_provenance_v4.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
