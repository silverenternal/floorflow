"""Record whether a matched RPLAN/full-CAD SOTA claim is currently testable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    cad = json.loads(
        (
            ROOT / "outputs/reports/final_gated_targetfree_cad_test503_v1.json"
        ).read_text(encoding="utf-8")
    )
    tlc_files = sorted(
        path.relative_to(ROOT / "third_party/TLC-Plan")
        for path in (ROOT / "third_party/TLC-Plan").rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    report = {
        "status": "blocked",
        "scope": "residential RPLAN and complete CAD generation",
        "targets": {
            "rplan_generation_sota": "blocked",
            "complete_cad_generation_sota": "blocked",
        },
        "sources_checked": {
            "FMLM": {
                "paper": "https://arxiv.org/abs/2604.04859",
                "project_page": "https://mapooon.github.io/FMLPage/",
                "finding": (
                    "Project page exposes paper/poster/video links but no "
                    "public model code or checkpoint link."
                ),
            },
            "TLC-Plan": {
                "paper": "https://arxiv.org/abs/2602.07100",
                "repository": "https://github.com/rosolose/TLC-PLAN",
                "commit": "8ed46c81cb3ca5b1cdff28cd36653153bb698b61",
                "tracked_non_git_files": [
                    str(path) for path in tlc_files
                ],
                "finding": (
                    "The repository currently contains only a README and an "
                    "image; the paper-stated code and trained models are absent."
                ),
            },
            "RPLAN": {
                "access": (
                    "https://docs.google.com/forms/d/e/"
                    "1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform"
                ),
                "finding": (
                    "Training data access is gated and local matched RPLAN "
                    "train/validation/test data are absent."
                ),
            },
            "CE2EPlan": {
                "audit": "docs/ce2eplan_audit_v1.md",
                "finding": (
                    "Local audit exists, but no FloorFlow-on-RPLAN matched run "
                    "or complete external preprocessing contract is available."
                ),
            },
        },
        "local_evidence": {
            "rplan_training_data_present": False,
            "floorflow_rplan_checkpoint_present": False,
            "executable_matched_baseline_present": False,
            "current_complete_cad_report": {
                "path": "outputs/reports/final_gated_targetfree_cad_test503_v1.json",
                "sha256": sha256(
                    ROOT / "outputs/reports/final_gated_targetfree_cad_test503_v1.json"
                ),
                "condition_mode": cad["condition_mode"],
                "cases": cad["cases"],
                "room_macro_iou": cad["room_metrics"]["overlay"][
                    "room_macro_iou"
                ],
                "fpeval_wall_f1": cad["room_metrics"]["overlay"][
                    "fpeval_wall_f1"
                ],
                "fpeval_opening_f1": cad["room_metrics"]["overlay"][
                    "fpeval_opening_f1"
                ],
                "fpeval_room_polygon_f1": cad["room_metrics"]["overlay"][
                    "fpeval_room_polygon_f1"
                ],
                "fpeval_edit_cost": cad["room_metrics"]["overlay"][
                    "fpeval_edit_cost"
                ],
            },
        },
        "checks": {
            "rplan_data_available": False,
            "floorflow_rplan_checkpoint_available": False,
            "public_baseline_code_available": False,
            "matched_preprocessing_available": False,
            "task_metrics_comparable": False,
        },
        "decision": (
            "Do not claim RPLAN or complete-CAD SOTA. The current complete-CAD "
            "evidence is a target-free CubiCasa diagnostic with substantially "
            "weaker geometric metrics than the oracle-control identity task."
        ),
        "unblock_requirements": [
            "Authorized RPLAN train/validation/test data.",
            "At least one executable public RPLAN baseline with released weights.",
            "A FloorFlow RPLAN model trained without test-set fitting.",
            "A predeclared task and metric contract shared by every method.",
        ],
    }
    output = ROOT / "outputs/reports/rplan_full_cad_sota_gate_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    document = ROOT / "docs/rplan_full_cad_claim_boundary_v1.md"
    document.write_text(
        "# RPLAN and Full-CAD Claim Boundary\n\nStatus: blocked.\n\nFMLM's page exposes no code or weight link. TLC-Plan's public repository currently contains only a README and one image. RPLAN training data remain gated.\n\nThe current 0.9718 CubiCasa layout-conditioned result is an oracle-control refinement task and is not comparable to RPLAN generation or complete-CAD leaderboards.\n\nA matched RPLAN/full-CAD SOTA claim requires authorized data, executable baselines, a FloorFlow RPLAN model, and one shared predeclared metric contract."
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
