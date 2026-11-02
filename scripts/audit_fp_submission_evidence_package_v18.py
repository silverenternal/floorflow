#!/usr/bin/env python3
"""Audit manifest, archive, and claim boundaries for v18."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

from floorflow.evaluation.deterministic_archive import (
    build_deterministic_tar_gz,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v18"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v18.tar.gz"
OUTPUT = ROOT / "outputs/reports/cvpr_vector_package_audit_v1.json"


def main() -> None:
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    expected = [row["path"] for row in manifest["files"]]
    disk = sorted(
        f"{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}"
        for path in PACKAGE.rglob("*")
        if path.is_file()
    )
    with tarfile.open(ARCHIVE, "r:gz") as handle:
        archived = sorted(handle.getnames())
    errors = []
    if expected != disk:
        errors.append("manifest files differ from package")
    if expected != archived:
        errors.append("manifest files differ from archive")
    artifact_errors = []
    for source, row in manifest["artifacts"].items():
        target = PACKAGE / row["package_path"]
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            artifact_errors.append(source)
    archive_hash = sha256_file(ARCHIVE)
    sidecar = ARCHIVE.with_name(f"{ARCHIVE.name}.sha256")
    declared = sidecar.read_text(encoding="utf-8").split()[0]
    with tempfile.TemporaryDirectory(prefix="fevp_v18_audit_") as temp:
        rebuilt = Path(temp) / ARCHIVE.name
        rebuilt_hash = build_deterministic_tar_gz(PACKAGE, rebuilt)
    claim = manifest["primary_claim"]
    boundaries = manifest["claim_boundaries"]
    checks = {
        "manifest_matches_package": expected == disk,
        "archive_matches_manifest": expected == archived,
        "artifact_hashes_current": not artifact_errors,
        "sidecar_matches": archive_hash == declared,
        "archive_deterministic": archive_hash == rebuilt_hash,
        "claim_boundaries_present": len(boundaries) >= 5,
        "no_housemind_defeat_phrase": "beats housemind" not in claim.lower(),
    }
    errors.extend(name for name, value in checks.items() if not value)
    audit = {
        "status": "complete" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "archive_sha256": archive_hash,
        "rebuilt_sha256": rebuilt_hash,
        "artifact_count": manifest["artifact_count"],
        "file_count": manifest["file_count"],
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
