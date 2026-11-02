#!/usr/bin/env python3
"""Build artifact provenance for the FEVP-2 vector evidence package."""

from __future__ import annotations

import json
from pathlib import Path

from build_cvpr_artifact_provenance_v1 import (
    _inputs_for,
    _override_generator,
    _script_candidates,
)
from cvpr_vector_artifact_registry import vector_canonical_artifacts

from floorflow.evaluation.deterministic_archive import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/reports/cvpr_artifact_provenance_v2.json"

VECTOR_GENERATORS = {
    "outputs/reports/vector_edit_benchmark_v1.json": (
        "scripts/build_vector_edit_benchmark_v1.py"
    ),
    "outputs/reports/fevp2_vector_attacks_v1.json": (
        "scripts/run_fevp2_vector_attacks_v1.py"
    ),
    "outputs/reports/vector_primitive_smoke_v1.json": (
        "scripts/build_vector_primitive_smoke_report_v1.py"
    ),
    "outputs/reports/vector_edit_baselines_v1.json": (
        "scripts/build_vector_edit_baselines_v1.py"
    ),
    "outputs/reports/vector_edit_results_v1.json": (
        "scripts/build_vector_edit_results_v1.py"
    ),
    "outputs/reports/vector_edit_significance_v1.json": (
        "scripts/build_vector_edit_significance_v1.py"
    ),
    "outputs/reports/vector_edit_strata_v1.json": (
        "scripts/build_vector_edit_strata_v1.py"
    ),
    "outputs/reports/vector_edit_failure_tail_v1.json": (
        "scripts/build_vector_edit_failure_pareto_v1.py"
    ),
    "outputs/reports/vector_edit_regression_gate_v1.json": (
        "scripts/build_vector_edit_regression_gate_v1.py"
    ),
    "outputs/reports/vector_residual_inference_v1.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_significance_v1.json": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "outputs/reports/vector_residual_inference_final_seed19.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_final_seed20.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_final_seed21.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_final_seed22.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_final_seed23.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch1.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch1_500.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch1_significance.json": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_500.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_seed20.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_seed21.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_seed22.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_seed23.json": (
        "scripts/evaluate_vector_residual_inference_v1.py"
    ),
    "outputs/reports/vector_residual_inference_full_epoch2_significance.json": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "outputs/reports/vector_residual_inference_multiseed_v1.json": (
        "scripts/build_vector_residual_inference_multiseed_v1.py"
    ),
    "outputs/reports/vector_topology_projection_v1.json": (
        "scripts/evaluate_vector_topology_projection_v1.py"
    ),
    "outputs/reports/vector_polygon_repair_v1.json": (
        "scripts/evaluate_vector_polygon_repair_v1.py"
    ),
    "outputs/reports/vector_polygon_repair_500_v1.json": (
        "scripts/evaluate_vector_polygon_repair_v1.py"
    ),
    "outputs/reports/vector_topology_factorial_v1.json": (
        "scripts/build_vector_topology_factorial_v1.py"
    ),
    "outputs/reports/vector_related_work_gate_v1.json": (
        "scripts/build_vector_related_work_gate_v1.py"
    ),
    "outputs/reports/cvpr_vector_claim_v1.json": (
        "scripts/build_cvpr_vector_claim_v1.py"
    ),
    "outputs/reports/cvpr_vector_claim_audit_v1.json": (
        "scripts/audit_cvpr_claim_text_v1.py"
    ),
    "paper/tables/vector_edit_main_results_v1.md": (
        "scripts/build_cvpr_vector_claim_v1.py"
    ),
    "paper/tables/vector_edit_strata_v1.md": (
        "scripts/build_vector_edit_strata_v1.py"
    ),
    "paper/tables/vector_edit_failure_tail_v1.md": (
        "scripts/build_vector_edit_failure_pareto_v1.py"
    ),
    "paper/tables/vector_residual_inference_v1.md": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "paper/tables/vector_residual_inference_full_epoch1.md": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "paper/tables/vector_residual_inference_full_epoch2.md": (
        "scripts/build_vector_residual_inference_significance_v1.py"
    ),
    "paper/tables/vector_residual_inference_multiseed_v1.md": (
        "scripts/build_vector_residual_inference_multiseed_v1.py"
    ),
    "outputs/reports/fp_line_limit_v18.json": (
        "scripts/audit_python_line_limit_v1.py"
    ),
}


def _generated(artifacts: list[str]) -> list[str]:
    return [
        path
        for path in artifacts
        if (
            path.startswith("outputs/reports/")
            and path.endswith(".json")
        )
        or path.startswith("paper/tables/")
        and path.endswith(".md")
    ]


def main() -> None:
    artifacts = vector_canonical_artifacts()
    generated = _generated(artifacts)
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    entries = {}
    missing = []
    for target in generated:
        relative = VECTOR_GENERATORS.get(target) or _override_generator(target)
        generator = ROOT / relative if relative else None
        if generator is None or not generator.is_file():
            candidates = _script_candidates(target, scripts)
            generator = candidates[0] if candidates else None
        if generator is None:
            missing.append(target)
            continue
        entries[target] = {
            "generator": str(generator.relative_to(ROOT)),
            "generator_sha256": sha256_file(generator),
            "inputs": _inputs_for(generator, target),
        }
    report = {
        "status": "complete" if not missing else "failed",
        "scope": (
            "Canonical v17 evidence plus FEVP-2 vector reports and tables."
        ),
        "generated_artifact_count": len(generated),
        "mapped_artifact_count": len(entries),
        "missing_artifacts": missing,
        "artifacts": entries,
        "hash_algorithm": "sha256",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "mapped": len(entries), "missing": missing}))
    raise SystemExit(0 if not missing else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
