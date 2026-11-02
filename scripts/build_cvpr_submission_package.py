#!/usr/bin/env python3
"""Assemble the frozen CVPR evidence and paper assets into one package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cvpr_package_v5_zero_human"),
    )
    parser.add_argument(
        "--package-version",
        default="cvpr_package_v5_zero_human",
    )
    parser.add_argument(
        "--cad-report",
        type=Path,
        default=Path(
            "outputs/reports/final_retrieval_free_style_test503_v1.json"
        ),
    )
    parser.add_argument(
        "--style-report",
        type=Path,
        default=Path(
            "outputs/reports/final_retrieval_free_style_val8_v1.json"
        ),
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    copy_path(Path("paper"), output / "paper")
    copy_path(Path("todo.json"), output / "todo.json")
    copy_path(
        Path("docs/cvpr_evidence_pack_v1.md"),
        output / "docs/cvpr_evidence_pack_v1.md",
    )
    copy_path(
        Path("docs/cvpr_quick_review_v1.md"),
        output / "docs/cvpr_quick_review_v1.md",
    )
    copy_path(
        Path("docs/external_baselines.md"),
        output / "docs/external_baselines.md",
    )
    copy_path(
        Path("docs/dynamic_head_provenance_v1.md"),
        output / "docs/dynamic_head_provenance_v1.md",
    )
    copy_path(
        Path("docs/recursive_generation.md"),
        output / "docs/recursive_generation.md",
    )
    copy_path(
        Path("docs/open_source_adaptation_v2.md"),
        output / "docs/open_source_adaptation_v2.md",
    )
    copy_path(
        Path("docs/cvpr_gap_v2.md"),
        output / "docs/cvpr_gap_v2.md",
    )
    copy_path(
        Path("docs/retrieval_free_prior_v1.md"),
        output / "docs/retrieval_free_prior_v1.md",
    )
    copy_path(
        Path("outputs/train/dynamic_furniture_queries_1024_v2/train.jsonl"),
        output / "reports/dynamic_furniture_queries_1024_v2_train.jsonl",
    )
    reports = [
        "style_design_gate_v1.json",
        "style_design_manifest_v1.json",
        "style_overlay_manifest_v1.json",
        "style_overlay_publication_gate_v1.json",
        "distribution_r94_overlay_all_styles_holdout300.json",
        "distribution_r94_overlay_all_styles_test503_w05.json",
        "distribution_r94_overlay_style_separation_val32_w05.json",
        "distribution_r94_seed23_style0_overlay_val64.json",
        "distribution_r94_style0_overlay_test503_w05.json",
        "external_baseline_comparison_v1.json",
        "model_sprint_screening_v1.json",
        "zero_human_model_iteration_v1.json",
        "zero_human_combination_matrix_v1.json",
        "dynamic_head_promotion_matrix_v1.json",
        "fixed_overlay_cad_complete_val64_v1.json",
        "final_targetfree_cad_test503_style0_v1.json",
        "final_adversarial_cad_test503_style0_v1.json",
        "adversarial_self_control_refiners_test503_v1.json",
        "target_free_contract_v1.json",
        "target_free_diagnostics_style0_test503_w05.json",
        "recursion_policy_v1.json",
        "recursive_room_segmenter_blend01_val491_v1.json",
        "recursive_cad_val491_v1.json",
        "recursive_cad_test503_v3.json",
        "inward_curriculum_audit_v1.json",
        "inward_curriculum_comparison_v1.json",
        "recursive_decision_audit_test503_v1.json",
        "recursive_style_stability_val8_v1.json",
        "final_gated_targetfree_cad_test503_v1.json",
        "final_gated_targetfree_cad_test503_per_case_v1.json",
        "final_gated_decision_audit_test503_v1.json",
        "final_gated_style_stability_val8_v1.json",
        "final_gated_diversity_val8x4_v1.json",
        "recursive_graph_adapter_1024_v1.json",
        "gated_room_selection_val64_v1.json",
        "inward_curriculum_comparison_v2.json",
        "metric_hierarchy_test503_v2.json",
        "model_inventory_v3.json",
        "b5_cvpr_readiness_v3.json",
        "retrieval_free_selection_val64_v1.json",
        "final_retrieval_free_selection_val64_v1.json",
        "merged_style_v2_candidate_val64_v1.json",
        "retrieval_free_cfm_val64_v2.json",
        "retrieval_free_trust_t035_val64_v1.json",
        "room_graph_trust_gate_smoke_v1.json",
        "geometry_cache_v384.json",
        "final_retrieval_free_style_test503_v1.json",
        "final_retrieval_free_style_per_case_test503_v1.json",
        "final_retrieval_free_style_decision_audit_test503_v1.json",
        "final_retrieval_free_style_val8_v1.json",
        "final_retrieval_free_diversity_val8x4_v1.json",
        "metric_hierarchy_test503_v3.json",
    ]
    for name in reports:
        copy_path(
            Path("outputs/reports") / name,
            output / "reports" / name,
        )
    copy_path(
        args.cad_report,
        output / "reports" / args.cad_report.name,
    )
    gate_path = output / "reports/style_overlay_publication_gate_v1.json"
    gate = (
        json.loads(gate_path.read_text(encoding="utf-8"))
        if gate_path.is_file()
        else {"status": "missing"}
    )
    external_path = (
        output / "reports/external_baseline_comparison_v1.json"
    )
    external = (
        json.loads(external_path.read_text(encoding="utf-8"))
        if external_path.is_file()
        else {"status": "missing"}
    )
    cad_path = output / "reports" / args.cad_report.name
    cad = (
        json.loads(cad_path.read_text(encoding="utf-8"))
        if cad_path.is_file()
        else {"cad_metrics": {}}
    )
    style = (
        json.loads(args.style_report.read_text(encoding="utf-8"))
        if args.style_report.is_file()
        else {}
    )
    manifest_files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    missing_metrics = list(
        cad["cad_metrics"].get("missing_metrics", [])
    )
    for metric in ("style_separation", "seed_iou_gap"):
        if metric in style and metric in missing_metrics:
            missing_metrics.remove(metric)
    checklist = {
        "package_version": args.package_version,
        "overlay_gate": gate["status"],
        "external_baselines": external["status"],
        "external_people_used": False,
        "manual_annotation_or_rating_used": False,
        "cad_hard_gate": cad["cad_metrics"].get("hard_gate", {}),
        "cad_missing_metrics": missing_metrics,
        "cad_balanced_secondary": cad["cad_metrics"].get("balanced_secondary", {}),
        "room_macro_iou_diagnostic": cad.get("room_metrics", {})
        .get("overlay", {})
        .get("room_macro_iou"),
        "room_iou_gate_met": (
            float(
                cad.get("room_metrics", {})
                .get("overlay", {})
                .get("room_macro_iou", 0.0)
            )
            >= 0.95
        ),
        "conditional_generation_gate_met": cad.get(
            "cad_metrics", {}
        ).get("hard_gate", {}).get("passed", False),
        "style_separation": style.get("style_separation"),
        "worst_style_room_iou": style.get("worst_style_room_iou"),
        "seed_iou_gap": style.get("seed_iou_gap"),
        "best_of_n_used": False,
        "target_derived_control_in_main_result": False,
        "python_file_limit": 600,
        "files": manifest_files,
    }
    (output / "submission_checklist.json").write_text(
        json.dumps(checklist, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: checklist[k] for k in checklist if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
