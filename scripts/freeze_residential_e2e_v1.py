"""Freeze validation-selected checkpoints, decode, thresholds, and claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/residential_e2e_validation_freeze_v1.json"
ARTIFACTS = (
    "outputs/reports/residential_e2e_multiseed_v1.json",
    "outputs/reports/residential_graph_refinement_exact_v1.json",
    "outputs/reports/residential_external_comparison_v1.json",
    "outputs/reports/residential_predicted_edit_v1.json",
    "outputs/reports/residential_predicted_refiner_v1.json",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoints = {
        "stage1": ROOT
        / "outputs/train/distribution_retrieval_free_merged_style_v2.pt",
        "predicted_refiner": ROOT
        / "outputs/train/residential_predicted_refiner_v1/best.pt",
        "raster2seq": ROOT
        / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth",
    }
    frozen = {
        "schema_version": "residential-e2e-validation-freeze-v1",
        "status": "frozen",
        "scope": "residential_floorplan_only",
        "selection_split": "validation_only",
        "test_opened": False,
        "checkpoints": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _hash(path),
                "bytes": path.stat().st_size,
            }
            for name, path in checkpoints.items()
        },
        "configuration": {
            "flow_steps": 12,
            "room_size": 224,
            "structure_size": 128,
            "graph_decode_sigma": 10.0,
            "graph_merge_min_area": 32,
            "graph_method": "direct_tiled",
            "matching_protocol": "type_geometry_hungarian",
            "raster2seq_renderer": "official_style",
            "single_sample": True,
            "best_of_n": False,
        },
        "claims": {
            "matched_cubicasa_reconstruction": "enabled",
            "corrected_predicted_completion": "enabled",
            "deterministic_graph_refinement": "enabled",
            "strict_locality_instruction_editing": "enabled_with_ged_limitation",
            "interaction_aware_quality": "disabled_unstable",
            "missing_control_watertightness": "disabled",
            "ood": "disabled",
            "rplan_sota": "disabled",
            "complete_cad_sota": "disabled",
            "house_mind_editing_defeat": "disabled",
            "free_text_sota": "disabled",
        },
        "validation_artifacts": {
            path: {"sha256": _hash(ROOT / path)}
            for path in ARTIFACTS
        },
    }
    OUTPUT.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze": str(OUTPUT), "checkpoints": frozen["checkpoints"]}, indent=2))


if __name__ == "__main__":
    main()
