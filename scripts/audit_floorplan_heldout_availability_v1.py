#!/usr/bin/env python3
"""Audit whether the current CubiCasa test split has a fresh held-out subset."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDIT_REPORT = ROOT / "outputs/reports/residential_instruction_robustness_test503_final_v1.json"
TEST_RASTERS = ROOT / "outputs/eval/fp_completion_predicted_refined_test503_final_v1/rasters"
TEST_MANIFEST = ROOT / "outputs/eval/fp_completion_predicted_refined_test503_final_v1/manifest.json"
OUTPUT = ROOT / "outputs/reports/floorplan_heldout_availability_v1.json"


def main() -> None:
    edit = json.loads(EDIT_REPORT.read_text(encoding="utf-8"))
    used = {row["name"] for row in edit["prompts"]}
    all_cases = {path.stem for path in TEST_RASTERS.glob("*.npz")}
    manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    report = {
        "schema_version": "floorplan-heldout-availability-v1",
        "status": "no_strictly_fresh_cubicasa_test_subset",
        "split": "test503",
        "edit_prompt_cases_used": len(used),
        "full_completion_cases_inferred": int(manifest["cases"]),
        "total_cases": len(all_cases),
        "case_ids_not_in_edit_prompts": len(all_cases - used),
        "strictly_fresh_heldout_available": False,
        "reason": (
            "The full 503-case test split was inferred by the predicted completion "
            "run, so a subset not appearing in the edit-prompt report is not a "
            "strictly unopened model split."
        ),
        "required_resolution": [
            "Acquire a new unused CubiCasa case manifest or external floorplan split.",
            "Freeze the model before any inference on the new split.",
            "Generate prompts from case IDs only, without inspecting outcomes.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
