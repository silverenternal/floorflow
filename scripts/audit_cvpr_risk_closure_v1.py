#!/usr/bin/env python3
"""Final source, provenance, manifest, and deterministic-archive audit."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tempfile
from pathlib import Path

from cvpr_evidence_registry import ROOT

from floorflow.evaluation.deterministic_archive import (
    build_deterministic_tar_gz,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=ROOT / "outputs/cvpr_package_fp_v17",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "outputs/cvpr_package_fp_v17.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cvpr_risk_closure_audit_v1.json",
    )
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unsupported_significance(value) -> list[str]:
    rows = []
    if isinstance(value, dict):
        for child in value.values():
            rows.extend(_unsupported_significance(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_unsupported_significance(child))
    elif (
        isinstance(value, str)
        and re.search(
            r"\bsignificant(?:ly)?\b",
            value,
            re.IGNORECASE,
        )
        and not re.search(
            r"\b(p\s*[<=>]|CI\b|confidence interval)",
            value,
            re.IGNORECASE,
        )
    ):
        rows.append(value)
    return rows


def _has_forbidden_housemind_claim(text: str) -> bool:
    pattern = re.compile(
        r"(outperform|beat|better than|superior to).{0,40}housemind",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        if not pattern.search(line):
            continue
        if re.search(r"\b(no|not|never|without)\b|不|未|无|没有", line):
            continue
        return True
    return False


def main() -> None:
    args = parse_args()
    closure = _read(ROOT / "outputs/reports/cvpr_risk_closure_v1.json")
    risks = _read(ROOT / "outputs/reports/fp_reviewer_risk_matrix_v9.json")
    provenance = _read(ROOT / "outputs/reports/cvpr_artifact_provenance_audit_v1.json")
    manifest = _read(args.package / "manifest.json")
    ledger = _read(ROOT / "outputs/reports/fp_cvpr_evidence_v12.json")
    claim_text = (ROOT / "paper/cvpr_claim_sections_v1.md").read_text(
        encoding="utf-8"
    )
    caption_text = (
        ROOT / "paper/tables/cvpr_table2_caption_v3.md"
    ).read_text(encoding="utf-8")
    writing_policy = (
        ROOT / "paper/writing_source_policy_v1.md"
    ).read_text(encoding="utf-8")
    errors = []
    if closure["status"] != "complete" or closure["resolved"] != closure["items"]:
        errors.append("source risk closure is incomplete")
    if risks["status"] != "complete" or risks["resolved"] != risks["rows"]:
        errors.append("reviewer risk matrix is incomplete")
    if provenance["status"] != "complete":
        errors.append("artifact provenance audit is incomplete")
    unsupported_significance = _unsupported_significance(ledger)
    if unsupported_significance:
        errors.append("canonical ledger has significance wording without p/CI")
    artifact_errors = []
    for source, row in manifest["artifacts"].items():
        target = args.package / row["package_path"]
        if not target.is_file():
            artifact_errors.append(f"missing packaged artifact: {source}")
            continue
        if sha256_file(target) != row["sha256"]:
            artifact_errors.append(f"packaged artifact hash mismatch: {source}")
    errors.extend(artifact_errors)
    manifest_files = manifest.get("files", [])
    expected_paths = [row["path"] for row in manifest_files]
    disk_paths = sorted(
        f"{args.package.name}/{path.relative_to(args.package).as_posix()}"
        for path in args.package.rglob("*")
        if path.is_file()
    )
    with tarfile.open(args.archive, "r:gz") as handle:
        tar_paths = sorted(handle.getnames())
    if expected_paths != disk_paths:
        errors.append("manifest files do not match package directory")
    if expected_paths != tar_paths:
        errors.append("manifest files do not match archive listing")
    if manifest.get("file_count") != len(expected_paths):
        errors.append("manifest file_count does not match files list")
    claim_sanity = {
        "no_housemind_outperformance_claim": not _has_forbidden_housemind_claim(
            claim_text
        ),
        "ce2eplan_generation_only_boundary": (
            "generation-only target" in claim_text
            and "not an editing-aware baseline" in claim_text
        ),
        "raster2seq_matched_reconstruction_boundary": (
            "matched reconstruction baseline" in claim_text
            and "not an editing-aware model" in claim_text
        ),
        "table2_v3_correction_record": (
            "Table 2" in caption_text
            and "Correction record (v3)" in caption_text
        ),
        "writing_source_policy_present": (
            "inside `outputs/cvpr_package_fp_v17.tar.gz`" in writing_policy
            and "fp_cvpr_evidence_v6.json" in writing_policy
        ),
    }
    errors.extend(
        f"submission sanity failed: {name}"
        for name, passed in claim_sanity.items()
        if not passed
    )
    lockfiles = {"pyproject.toml", "uv.lock", ".python-version"}
    if not lockfiles <= set(manifest["artifacts"]):
        errors.append("lockfiles are absent from the manifest")
    archive_hash = sha256_file(args.archive)
    sidecar = args.archive.with_name(f"{args.archive.name}.sha256")
    declared_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if archive_hash != declared_hash:
        errors.append("archive sidecar hash mismatch")
    with tempfile.TemporaryDirectory(prefix="fevp_v17_rebuild_") as temp:
        rebuilt = Path(temp) / args.archive.name
        rebuilt_hash = build_deterministic_tar_gz(args.package, rebuilt)
    if rebuilt_hash != archive_hash:
        errors.append("independent archive rebuild hash mismatch")
    audit = {
        "status": "complete" if not errors else "failed",
        "source_risk_checks": {
            "total": closure["items"],
            "resolved": closure["resolved"],
        },
        "reviewer_risks": {
            "total": risks["rows"],
            "resolved": risks["resolved"],
        },
        "provenance": provenance,
        "package": {
            "path": str(args.package.relative_to(ROOT)),
            "status": manifest["status"],
            "artifact_count": manifest["artifact_count"],
            "file_count": manifest["file_count"],
            "archive_sha256": archive_hash,
            "independent_rebuild_sha256": rebuilt_hash,
            "deterministic": rebuilt_hash == archive_hash,
        },
        "errors": errors,
        "checks": {
            "all_numbered_source_checks_resolved": (
                closure["resolved"] == closure["items"]
            ),
            "all_reviewer_risks_resolved": risks["resolved"] == risks["rows"],
            "provenance_complete": provenance["status"] == "complete",
            "canonical_significance_bounded": not unsupported_significance,
            "manifest_files_match_package": expected_paths == disk_paths,
            "archive_listing_matches_manifest": expected_paths == tar_paths,
            "all_packaged_hashes_current": not artifact_errors,
            "lockfiles_in_package": lockfiles <= set(manifest["artifacts"]),
            "archive_sidecar_matches": archive_hash == declared_hash,
            "archive_rebuild_is_deterministic": rebuilt_hash == archive_hash,
        },
        "submission_sanity": claim_sanity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
