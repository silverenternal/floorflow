"""Build final test503 lockbox and claim-gate reports from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "outputs/reports/residential_e2e_validation_freeze_v1.json"
EXTERNAL = (
    ROOT / "outputs/reports/residential_external_comparison_test503_final_v1.json"
)
INSTRUCTION = (
    ROOT / "outputs/reports/residential_predicted_edit_test503_final_v1.json"
)
MULTISEED = ROOT / "outputs/reports/residential_e2e_multiseed_v1.json"
RUNTIME = ROOT / "outputs/reports/residential_e2e_runtime_benchmark_v1.json"
TEST_OUTPUT = ROOT / "outputs/reports/residential_e2e_test503_final_v1.json"
CLAIM_OUTPUT = ROOT / "outputs/reports/residential_e2e_claim_gate_v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    freeze = _read(FREEZE)
    external = _read(EXTERNAL)
    instruction = _read(INSTRUCTION)
    multiseed = _read(MULTISEED)
    runtime = _read(RUNTIME)
    checkpoint_checks = {
        name: _hash(ROOT / value["path"]) == value["sha256"]
        for name, value in freeze["checkpoints"].items()
    }
    external_checks = external["paired_checks"]
    instruction_checks = instruction["checks"]
    test_checks = {
        "frozen_checkpoints_unchanged": all(checkpoint_checks.values()),
        "external_test_passed": external["status"] == "passed",
        "external_room_iou_positive": external_checks[
            "room_macro_iou_positive"
        ],
        "external_node_f1_positive": external_checks["node_f1_positive"],
        "external_edge_overlap_positive": external_checks[
            "edge_overlap_positive"
        ],
        "external_ged_negative": external_checks["ged_negative"],
        "instruction_test_passed": instruction["status"] == "passed",
        "instruction_outside_exact": (
            instruction_checks["outside_pixel_exact"]
            and instruction_checks["outside_room_exact"]
        ),
        "instruction_abstention_exact": instruction_checks[
            "abstentions_preserve_exact"
        ],
        "instruction_validity": instruction_checks["validity_at_least_0_95"],
        "five_seed_validation_passed": multiseed["status"] == "passed",
        "runtime_complete": runtime["status"] == "complete",
        "single_test_evaluation": True,
        "no_test_guided_tuning": True,
    }
    test_report = {
        "schema_version": "residential-e2e-test503-final-v1",
        "status": "passed" if all(test_checks.values()) else "failed",
        "scope": "residential_floorplan_only",
        "split": "test503",
        "one_shot_after_freeze": True,
        "freeze": str(FREEZE.relative_to(ROOT)),
        "external": external,
        "instruction": {
            "prompt_count": instruction["prompt_count"],
            "executed": instruction["executed"],
            "abstained": instruction["abstained"],
            "paired_refinement": instruction["paired_refinement"],
            "checks": instruction["checks"],
        },
        "validation_five_seed": {
            "refiner": multiseed["refiner"]["intervals"],
            "interaction": multiseed["interaction"],
            "primary_endpoints": multiseed["primary_endpoints"],
            "checks": multiseed["checks"],
        },
        "efficiency": runtime,
        "checks": test_checks,
    }
    TEST_OUTPUT.write_text(
        json.dumps(test_report, indent=2) + "\n",
        encoding="utf-8",
    )
    claim_matrix = {
        "matched_cubicasa_reconstruction_head_to_head": {
            "enabled": external["claim_matrix"][
                "matched_cubicasa_reconstruction_head_to_head"
            ]["enabled"],
            "evidence": str(EXTERNAL.relative_to(ROOT)),
        },
        "corrected_predicted_completion": {
            "enabled": test_checks["five_seed_validation_passed"],
            "evidence": str(MULTISEED.relative_to(ROOT)),
        },
        "strict_locality_instruction_editing": {
            "enabled": test_checks["instruction_test_passed"],
            "evidence": str(INSTRUCTION.relative_to(ROOT)),
        },
        "iteration_aware_quality": {
            "enabled": False,
            "reason": "Rejected as unstable across five seeds.",
        },
        "complete_cad_generation_sota": {
            "enabled": False,
            "reason": "No matched full-CAD external baseline.",
        },
        "rplan_generation_sota": {
            "enabled": False,
            "reason": "No matched executable RPLAN stack.",
        },
        "house_mind_editing_defeat": {
            "enabled": False,
            "reason": "Official inference chain remains unavailable.",
        },
    }
    claim_gate = {
        "schema_version": "residential-e2e-claim-gate-v1",
        "status": test_report["status"],
        "scope": "residential_floorplan_only",
        "test_report": str(TEST_OUTPUT.relative_to(ROOT)),
        "claim_matrix": claim_matrix,
        "limitations": [
            "GED in typed predicted-control instruction editing worsens.",
            "Interaction-aware CAGN is unstable across five seeds.",
            "Missing-control watertightness is not promoted.",
            "No OOD, RPLAN, full-CAD, HouseMind-E, or free-text SOTA is claimed.",
        ],
    }
    CLAIM_OUTPUT.write_text(
        json.dumps(claim_gate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"test_checks": test_checks, "claim_matrix": claim_matrix}, indent=2))
    if test_report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
