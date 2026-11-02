#!/usr/bin/env python3
"""Build artifact-to-generator-to-input provenance for canonical evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cvpr_evidence_registry import ROOT, canonical_artifacts

from floorflow.evaluation.deterministic_archive import sha256_file

PROVENANCE_OUTPUT = "outputs/reports/cvpr_artifact_provenance_v1.json"
PATH_PATTERN = re.compile(
    r"[\"']((?:configs|examples|outputs|schema|paper|src|scripts)/"
    r"[^\"'\\n]+)[\"']"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / PROVENANCE_OUTPUT,
    )
    return parser.parse_args()


def _generated_artifacts(artifacts: list[str]) -> list[str]:
    return [
        path
        for path in artifacts
        if (
            path.startswith("outputs/reports/")
            and path.endswith(".json")
        )
        or path.startswith("outputs/external_baselines/")
        and path.endswith("/report.json")
        or path.startswith("paper/tables/")
        and path.endswith(".md")
    ]


def _script_candidates(target: str, scripts: list[Path]) -> list[Path]:
    basename = Path(target).name
    target_stem = Path(target).stem
    candidates = []
    for script in scripts:
        if script.name in {
            "build_cvpr_artifact_provenance_v1.py",
            "audit_cvpr_artifact_provenance_v1.py",
        }:
            continue
        text = script.read_text(encoding="utf-8")
        score = 0
        if basename in text:
            score += 100
        if target_stem in script.stem:
            score += 50
        if script.stem in target_stem:
            score += 25
        if score:
            candidates.append((score, script))
    return [
        script
        for _, script in sorted(
            candidates,
            key=lambda item: (-item[0], str(item[1])),
        )
    ]


def _override_generator(target: str) -> str | None:
    exact = {
        "outputs/reports/cvpr_claim_text_audit_v1.json": (
            "scripts/audit_cvpr_claim_text_v1.py"
        ),
        "outputs/reports/cvpr_risk_closure_v1.json": (
            "scripts/build_cvpr_risk_closure_v1.py"
        ),
        "outputs/reports/cvpr_risk_closure_audit_v1.json": (
            "scripts/audit_cvpr_risk_closure_v1.py"
        ),
        "outputs/reports/cvpr_artifact_provenance_v1.json": (
            "scripts/build_cvpr_artifact_provenance_v1.py"
        ),
        "outputs/reports/cvpr_artifact_provenance_audit_v1.json": (
            "scripts/audit_cvpr_artifact_provenance_v1.py"
        ),
        "outputs/reports/fp_cvpr_evidence_v12.json": (
            "scripts/build_fp_evidence_v12.py"
        ),
        "outputs/reports/fp_reviewer_risk_matrix_v9.json": (
            "scripts/build_reviewer_risk_matrix_v9.py"
        ),
        "outputs/reports/fevp_budget_matched_control_v2.json": (
            "scripts/build_fevp_budget_matched_control_v2.py"
        ),
        "outputs/reports/fevp_seed_attribution_v2.json": (
            "scripts/build_fevp_seed_attribution_v2.py"
        ),
        "outputs/reports/external_baseline_protocol_audit_v2.json": (
            "scripts/build_external_baseline_protocol_audit_v2.py"
        ),
        "outputs/reports/fevp_attack_supplement_v1.json": (
            "scripts/build_fevp_attack_supplement_v1.py"
        ),
        "paper/tables/cvpr_table2_caption_v3.md": (
            "scripts/build_cvpr_table2_caption_v1.py"
        ),
        "outputs/reports/fp_line_limit_v17.json": (
            "scripts/audit_python_line_limit_v1.py"
        ),
    }
    if target in exact:
        return exact[target]
    if target.startswith(
        "outputs/reports/fp_edit_detection_test503_controlgated_seed"
    ):
        return "scripts/build_fp_edit_detection_multiseed_report.py"
    if target.startswith(
        "outputs/reports/fp_delta_mse_decomposition_controlgated_test503_seed"
    ):
        return "scripts/build_fp_delta_mse_multiseed_report.py"
    if target.startswith("outputs/reports/fp_structural_error_strata_final_seed"):
        return "scripts/build_fp_canonical_structural_multiseed_report.py"
    return None


def _inputs_for(generator: Path, target: str) -> dict[str, str]:
    text = generator.read_text(encoding="utf-8")
    inputs = {}
    for match in PATH_PATTERN.findall(text):
        path = ROOT / match
        relative = str(path.relative_to(ROOT))
        if not path.is_file() or relative == target:
            continue
        inputs[relative] = sha256_file(path)
    return dict(sorted(inputs.items()))


def main() -> None:
    args = parse_args()
    artifacts = canonical_artifacts()
    generated = _generated_artifacts(artifacts)
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    entries = {}
    missing = []
    for target in generated:
        override = _override_generator(target)
        generator = ROOT / override if override else None
        if generator is None:
            candidates = _script_candidates(target, scripts)
            generator = candidates[0] if candidates else None
        if generator is None or not generator.is_file():
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
            "Every canonical output/report JSON, external report JSON, and "
            "paper/tables Markdown artifact."
        ),
        "generated_artifact_count": len(generated),
        "mapped_artifact_count": len(entries),
        "missing_artifacts": missing,
        "hash_algorithm": "sha256",
        "artifacts": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "mapped": len(entries),
                "missing": missing,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
