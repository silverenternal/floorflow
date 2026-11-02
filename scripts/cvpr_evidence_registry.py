"""Canonical artifact registry for the CVPR risk-closure package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_DOCS = {
    "docs/fevp_final_claim_v2.md",
    "docs/fevp_final_claim_v3.md",
    "docs/fp_model_review_and_sota_gate_v1.md",
}
HISTORICAL_REPORTS = {
    *(f"outputs/reports/fp_cvpr_evidence_v{index}.json" for index in range(1, 12)),
    *(
        f"outputs/reports/fp_reviewer_risk_matrix_v{index}.md"
        for index in range(1, 9)
    ),
    "outputs/reports/fevp_final_claim_v2.json",
    "outputs/reports/fevp_final_claim_v3.json",
}
CANONICAL_DOCS = {
    "docs/ce2eplan_audit_v1.md",
    "docs/fevp_claim_language_v1.md",
    "docs/fevp_final_claim_v4.md",
    "docs/fevp_formal_spec_v3.md",
    "docs/fevp_housemind_positioning_v2.md",
    "docs/fevp_metric_contracts_v2.md",
    "docs/fevp_v1_0.md",
    "docs/fp_cvpr2026_positioning_evidence_v1.md",
    "docs/fp_locality_mechanism_claims_v1.md",
    "docs/fp_strict_locality_protocol_v2.md",
}

CLOSURE_ARTIFACTS = (
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "configs/fp_raster2seq_requirements.txt",
    "src/floorflow/evaluation/statistical_effect_sizes.py",
    "src/floorflow/evaluation/deterministic_archive.py",
    "tests/test_statistical_effect_sizes.py",
    "tests/test_deterministic_archive.py",
    "scripts/cvpr_evidence_registry.py",
    "scripts/build_fevp_budget_matched_control_v2.py",
    "scripts/build_fevp_seed_attribution_v2.py",
    "scripts/build_external_baseline_protocol_audit_v2.py",
    "scripts/build_fevp_attack_supplement_v1.py",
    "scripts/build_fevp_fiveseed_stability_v2.py",
    "scripts/build_fevp_edit_strength_v2.py",
    "scripts/build_fevp_graph_metrics_v2.py",
    "scripts/build_fevp_adversarial_prompts_v3.py",
    "scripts/build_cvpr_table2_caption_v1.py",
    "scripts/audit_cvpr_claim_text_v1.py",
    "scripts/build_cvpr_risk_closure_v1.py",
    "scripts/audit_cvpr_risk_closure_v1.py",
    "scripts/build_cvpr_artifact_provenance_v1.py",
    "scripts/audit_cvpr_artifact_provenance_v1.py",
    "scripts/audit_python_line_limit_v1.py",
    "scripts/build_fp_evidence_v12.py",
    "scripts/build_reviewer_risk_matrix_v9.py",
    "scripts/build_fp_submission_evidence_package_v17.py",
    "outputs/reports/fevp_budget_matched_control_v2.json",
    "outputs/reports/fevp_seed_attribution_v2.json",
    "outputs/reports/external_baseline_protocol_audit_v2.json",
    "outputs/reports/fevp_attack_supplement_v1.json",
    "outputs/reports/fevp_fiveseed_stability_v2.json",
    "outputs/reports/fevp_edit_strength_v2.json",
    "outputs/reports/fevp_graph_metrics_v2.json",
    "outputs/reports/fevp_adversarial_prompts_v3.json",
    "outputs/reports/cvpr_claim_text_audit_v1.json",
    "outputs/reports/cvpr_risk_closure_v1.json",
    "outputs/reports/cvpr_risk_closure_v1.md",
    "outputs/reports/cvpr_artifact_provenance_v1.json",
    "outputs/reports/cvpr_artifact_provenance_audit_v1.json",
    "outputs/reports/fp_cvpr_evidence_v12.json",
    "outputs/reports/fp_reviewer_risk_matrix_v9.json",
    "outputs/reports/fp_reviewer_risk_matrix_v9.md",
    "outputs/reports/fp_pytest_v17.xml",
    "outputs/reports/fp_ruff_v17.txt",
    "outputs/reports/fp_line_limit_v17.json",
    "paper/cvpr_claim_sections_v1.md",
    "paper/writing_source_policy_v1.md",
    "paper/tables/cvpr_table2_caption_v3.md",
    "docs/fevp_formal_spec_v3.md",
)


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prior_artifacts() -> list[str]:
    v16 = load(
        "registry_v16",
        "scripts/build_fp_submission_evidence_package_v16.py",
    )
    v15 = load(
        "registry_v15",
        "scripts/build_fp_submission_evidence_package_v15.py",
    )
    v14 = load(
        "registry_v14",
        "scripts/build_fp_submission_evidence_package_v14.py",
    )
    v13 = load(
        "registry_v13",
        "scripts/build_fp_submission_evidence_package_v13.py",
    )
    v12 = load(
        "registry_v12",
        "scripts/build_fp_submission_evidence_package_v12.py",
    )
    v11 = v12.load_v11_builder()
    return list(
        dict.fromkeys(
            [
                *v11.ARTIFACTS,
                *v12.extra_artifacts(),
                *v13._hardened_artifacts(),
                *v14._offensive_artifacts(),
                *v15._auditability_artifacts(),
                *v16._portability_artifacts(),
                *CLOSURE_ARTIFACTS,
            ]
        )
    )


def canonical_artifacts() -> list[str]:
    """Return sorted, curated artifacts and reject stale narrative documents."""

    values = []
    for path in _prior_artifacts():
        if path in HISTORICAL_DOCS or path in HISTORICAL_REPORTS:
            continue
        if path.startswith("docs/") and path not in CANONICAL_DOCS:
            continue
        values.append(path)
    return sorted(dict.fromkeys(values))


def data_directories() -> list[str]:
    v12 = load(
        "registry_data_v12",
        "scripts/build_fp_submission_evidence_package_v12.py",
    )
    v11 = v12.load_v11_builder()
    return [
        *v11.DATA_DIRECTORIES,
        "outputs/external_baselines/raster2seq/official_test509_v1/jsons",
    ]


__all__ = [
    "CANONICAL_DOCS",
    "CLOSURE_ARTIFACTS",
    "ROOT",
    "canonical_artifacts",
    "data_directories",
]
