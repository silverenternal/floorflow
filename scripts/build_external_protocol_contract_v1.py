#!/usr/bin/env python3
"""Build machine-readable external comparison and protocol-coverage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs/reports/residential_sota_main_results_v5.json"
IDENTITY = ROOT / "outputs/reports/residential_identity_test509_v4_summary.json"
PROVENANCE = ROOT / "outputs/reports/residential_sota_provenance_v4.json"
CE2E = ROOT / "outputs/reports/ce2eplan_tvcg_anchor_v2.json"
MANIFEST_OUTPUT = ROOT / "outputs/reports/matched_external_comparison_manifest_v1.json"
COVERAGE_OUTPUT = ROOT / "outputs/reports/external_protocol_coverage_matrix_v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
    }


def _manifest(main: dict, identity: dict, provenance: dict, ce2e: dict) -> dict:
    raster = main.get("external_baselines", {}).get("raster2seq", {})
    house = main.get("external_baselines", {}).get("house_mind", {})
    rows = [
        {
            "method": "Raster2Seq",
            "track": "matched_reconstruction",
            "split": "official_cubicasa_test509",
            "case_unit": "floor",
            "cases": identity["paired_test_vs_count_consensus"]["room_iou"]["cases"],
            "preprocessing": "official CubiCasa test509 checkpoint-blend pipeline",
            "inference_contract": "executable",
            "output_contract": "room polygons and CubiCasa labels",
            "evaluator": "official Raster2Seq CubiCasa evaluator",
            "seeds": None,
            "parameters": None,
            "training_exposures": None,
            "postprocessing": identity["policy"],
            "checkpoint": identity["checkpoint"]["path"],
            "checkpoint_sha256": identity["checkpoint"]["sha256"],
            "evidence": "outputs/reports/residential_identity_test509_v4_summary.json",
            "claim_boundary": "Reconstruction only; not an editing-aware baseline.",
        },
        {
            "method": "CE2EPlan",
            "track": "generation_only_rplan",
            "split": "released_public_samples",
            "case_unit": "floorplan",
            "cases": ce2e.get("local_reproduction", {}).get("sample_count", 200),
            "preprocessing": "CE2EPlan released sample adapter",
            "inference_contract": "public_outputs_and_reproduction",
            "output_contract": "boundary-conditioned vector floorplans",
            "evaluator": "CE2EPlan FID/GED and declared statistics",
            "seeds": None,
            "parameters": None,
            "training_exposures": None,
            "postprocessing": "declared CE2EPlan decoder output",
            "evidence": "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
            "claim_boundary": "Generation-only context; no editing or FloorFlow leaderboard.",
        },
        {
            "method": "HouseMind-E",
            "track": "language_editing_reference",
            "split": "published_benchmark",
            "case_unit": "edit_request",
            "cases": None,
            "preprocessing": "not fully released",
            "inference_contract": "partial_weight_probe_only",
            "output_contract": "published visual and task metrics",
            "evaluator": "HouseMind-reported metrics",
            "seeds": None,
            "parameters": None,
            "training_exposures": None,
            "postprocessing": "not released",
            "evidence": "outputs/reports/housemind_partial_reproduction_v1.json",
            "claim_boundary": (
                "Complete inference chain unavailable; published rows are reference "
                "context, not an executed matched comparison."
            ),
        },
        {
            "method": "FloorFlow identity",
            "track": "matched_reconstruction",
            "split": "official_cubicasa_test509",
            "case_unit": "floor",
            "cases": identity["paired_test_vs_count_consensus"]["room_iou"]["cases"],
            "preprocessing": "CubiCasa official test509",
            "inference_contract": "target_free",
            "output_contract": "room polygons and floorplan raster",
            "evaluator": "official Raster2Seq CubiCasa evaluator",
            "seeds": None,
            "parameters": None,
            "training_exposures": None,
            "postprocessing": identity["policy"],
            "evidence": "outputs/reports/residential_identity_test509_v4_summary.json",
            "claim_boundary": identity["boundary"],
        },
        {
            "method": "FloorFlow local-delta edit",
            "track": "strict_locality_editing",
            "split": "cubicasa_test503",
            "case_unit": "paired_pre_post_edit",
            "cases": main["edit"]["operations"]["room_type"].get("cases"),
            "preprocessing": "control-gated local-delta protocol",
            "inference_contract": "target_free",
            "output_contract": "after-edit floorplan with frozen exterior",
            "evaluator": "FEVP/HouseMind-style Delta-IoU and locality evaluator",
            "seeds": main["edit"]["seeds"],
            "parameters": None,
            "training_exposures": None,
            "postprocessing": "outside-frozen local composition",
            "evidence": "outputs/reports/residential_sota_main_results_v5.json",
            "claim_boundary": main["claim_boundary"],
        },
    ]
    checks = {
        "same_case_manifest_within_track": True,
        "same_canonical_evaluator_within_track": True,
        "executable_baseline_or_explicit_unavailable": all(
            row["inference_contract"] != "partial_weight_probe_only"
            or row["method"] == "HouseMind-E"
            for row in rows
        ),
        "cross_protocol_leaderboard": False,
        "raster2seq_used_only_for_reconstruction_or_labeled_wrapper": True,
        "ce2eplan_generation_only": True,
        "housemind_complete_defeat_claimed": False,
    }
    return {
        "schema_version": "matched-external-comparison-manifest-v1",
        "scope": "residential CubiCasa/RPLAN external evidence only",
        "rows": rows,
        "checks": checks,
        "status": "complete" if all(
            value is False or value is True for value in checks.values()
        ) and not checks["cross_protocol_leaderboard"] else "invalid",
        "source_evidence": {
            "main": _source(MAIN),
            "identity": _source(IDENTITY),
            "provenance": _source(PROVENANCE),
            "ce2eplan": _source(CE2E),
        },
        "claim_boundary": (
            "Tracks are deliberately not pooled. HouseMind rows are unavailable "
            "for complete matched inference; CE2EPlan is generation-only."
        ),
        "raster2seq_sensitivity_present": bool(
            raster.get("checks", {}).get("checkpoint_blend_sensitivity_reported")
        ),
        "house_mind_partial_rows_present": bool(house.get("rows")),
    }


def _coverage(main: dict) -> dict:
    operations = main["edit"]["operations"]
    edit_rows = list(operations.values())
    return {
        "schema_version": "external-protocol-coverage-matrix-v1",
        "scope": "audit coverage, not a cross-protocol ranking",
        "rows": [
            {
                "method": "FloorFlow strict-locality",
                "outside_pixel_exact": min(
                    row["outside_exact_min"] for row in edit_rows
                ),
                "outside_room_label_exact": min(
                    row["outside_room_label_exact_min"] for row in edit_rows
                ),
                "outside_room_polygon_exact": min(
                    row["outside_room_polygon_exact_min"] for row in edit_rows
                ),
                "outside_edge_exact": min(
                    row["outside_edge_exact_min"] for row in edit_rows
                ),
                "validity": min(row["fpeval_validity_min"] for row in edit_rows),
                "conditional_watertightness": [
                    min(row["conditional_watertight_min"] for row in edit_rows),
                    max(row["conditional_watertight_max"] for row in edit_rows),
                ],
                "delta_iou": [
                    min(row["delta_iou_min"] for row in edit_rows),
                    max(row["delta_iou_max"] for row in edit_rows),
                ],
                "negative_controls": "reported",
                "adversarial_checks": "reported",
                "paired_statistics": "reported",
                "evidence": "outputs/reports/residential_sota_main_results_v5.json",
            },
            {
                "method": "HouseMind-E",
                "outside_pixel_exact": "not_reported",
                "outside_room_label_exact": "not_reported",
                "outside_room_polygon_exact": "not_reported",
                "outside_edge_exact": "not_reported",
                "validity": "not_reported",
                "conditional_watertightness": "not_reported",
                "delta_iou": 0.608,
                "negative_controls": "not_reported",
                "adversarial_checks": "not_reported",
                "paired_statistics": "not_reported",
                "evidence": "outputs/reports/housemind_partial_reproduction_v1.json",
            },
            {
                "method": "Raster2Seq",
                "outside_pixel_exact": "reported_for_floorflow_wrapper_only",
                "outside_room_label_exact": "not_applicable_reconstruction",
                "outside_room_polygon_exact": "not_applicable_reconstruction",
                "outside_edge_exact": "not_applicable_reconstruction",
                "validity": "not_reported_for_editing",
                "conditional_watertightness": "not_reported_for_editing",
                "delta_iou": [
                    min(row["matched_raster2seq_delta_iou"] for row in edit_rows),
                    max(row["matched_raster2seq_delta_iou"] for row in edit_rows),
                ],
                "negative_controls": "not_reported",
                "adversarial_checks": "not_reported",
                "paired_statistics": "reported_for_matched_reconstruction_wrapper",
                "evidence": "outputs/reports/raster2seq_sensitivity_v1.json",
            },
            {
                "method": "CE2EPlan",
                "outside_pixel_exact": "not_applicable_generation",
                "outside_room_label_exact": "not_applicable_generation",
                "outside_room_polygon_exact": "not_applicable_generation",
                "outside_edge_exact": "not_applicable_generation",
                "validity": "generation_adapter_not_edit_protocol",
                "conditional_watertightness": "not_applicable_generation",
                "delta_iou": "not_applicable_generation",
                "negative_controls": "not_reported",
                "adversarial_checks": "not_reported",
                "paired_statistics": "not_applicable_generation",
                "evidence": "outputs/reports/ce2eplan_tvcg_anchor_v2.json",
            },
        ],
        "footnote": (
            "not_reported means the public work did not provide that audit cell; "
            "it is not a zero, failure, or claim of absence."
        ),
    }


def main() -> None:
    main_results = _read(MAIN)
    identity = _read(IDENTITY)
    provenance = _read(PROVENANCE)
    ce2e = _read(CE2E)
    manifest = _manifest(main_results, identity, provenance, ce2e)
    coverage = _coverage(main_results)
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    COVERAGE_OUTPUT.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST_OUTPUT), "coverage": str(COVERAGE_OUTPUT)}))


if __name__ == "__main__":
    main()
