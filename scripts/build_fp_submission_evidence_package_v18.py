#!/usr/bin/env python3
"""Build the deterministic FEVP-2 CVPR evidence package v18."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from cvpr_evidence_registry import data_directories
from cvpr_vector_artifact_registry import vector_canonical_artifacts

from floorflow.evaluation.deterministic_archive import (
    build_deterministic_tar_gz,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "outputs/cvpr_package_fp_v18"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v18.tar.gz"
EXTRAS = (
    "scripts/cvpr_vector_artifact_registry.py",
    "scripts/build_cvpr_artifact_provenance_v2.py",
    "scripts/audit_cvpr_artifact_provenance_v2.py",
    "scripts/build_fp_evidence_v13.py",
    "scripts/build_reviewer_risk_matrix_v10.py",
    "scripts/build_fp_submission_evidence_package_v18.py",
    "outputs/reports/cvpr_artifact_provenance_v2.json",
    "outputs/reports/cvpr_artifact_provenance_audit_v2.json",
    "outputs/reports/fp_cvpr_evidence_v13.json",
    "outputs/reports/fp_reviewer_risk_matrix_v10.json",
    "outputs/reports/fp_reviewer_risk_matrix_v10.md",
    "outputs/reports/fp_pytest_v18.xml",
    "outputs/reports/fp_ruff_v18.txt",
    "outputs/reports/fp_line_limit_v18.json",
)


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _pytest(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("pytest report has no testsuite")
    return {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }


def _directory_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file())


def _directory_digest(path: Path) -> tuple[int, int, str]:
    value = hashlib.sha256()
    files = _directory_files(path)
    for item in files:
        value.update(str(item.relative_to(path)).encode("utf-8"))
        value.update(b"\0")
        value.update(sha256_file(item).encode("ascii"))
    return len(files), sum(item.stat().st_size for item in files), value.hexdigest()


def _created_at() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def main() -> None:
    artifacts = sorted(dict.fromkeys([*vector_canonical_artifacts(), *EXTRAS]))
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v18 artifacts: {missing}")
    evidence = _read("outputs/reports/fp_cvpr_evidence_v13.json")
    claim = _read("outputs/reports/cvpr_vector_claim_v1.json")
    risks = _read("outputs/reports/fp_reviewer_risk_matrix_v10.json")
    provenance = _read("outputs/reports/cvpr_artifact_provenance_v2.json")
    pytest_report = ROOT / "outputs/reports/fp_pytest_v18.xml"
    ruff_report = ROOT / "outputs/reports/fp_ruff_v18.txt"
    artifact_rows = {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256_file(ROOT / path),
            "package_path": f"artifacts/{path}",
        }
        for path in artifacts
    }
    directory_rows = {}
    data_files = []
    for relative in data_directories():
        source = ROOT / relative
        files, size, tree_hash = _directory_digest(source)
        directory_rows[relative] = {
            "files": files,
            "bytes": size,
            "tree_sha256": tree_hash,
            "package_path": f"data/{relative}",
        }
        for item in _directory_files(source):
            shipped = item.relative_to(ROOT).as_posix()
            data_files.append(
                {
                    "path": f"{PACKAGE.name}/data/{shipped}",
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    readme = f"""# FloorFlow FEVP-2 CVPR Experiment Evidence v18

This package contains experiment evidence only; no manuscript is included.

## Claim

{claim['claim']}

## Verification

- Vector cases: {claim['metadata']['cases']}
- Reviewer risks resolved: {risks['resolved']}/{risks['rows']}
- Provenance mappings: {provenance['mapped_artifact_count']}
- Deterministic archive hash is recorded in the sibling `.sha256` file.

The package preserves the v17 raster evidence while adding the FEVP-2 vector
benchmark, five-seed results, matched Raster2Seq audit, failure tails, figure,
and reviewer boundaries.
"""
    readme_bytes = readme.encode("utf-8")
    files = [
        {
            "path": f"{PACKAGE.name}/artifacts/{path}",
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for path, row in artifact_rows.items()
    ]
    files.extend(data_files)
    files.append(
        {
            "path": f"{PACKAGE.name}/README.md",
            "bytes": len(readme_bytes),
            "sha256": hashlib.sha256(readme_bytes).hexdigest(),
        }
    )
    files.append(
        {
            "path": f"{PACKAGE.name}/manifest.json",
            "self_reference": True,
        }
    )
    files.sort(key=lambda row: row["path"])
    manifest = {
        "status": "fevp2_vector_evidence_v18",
        "created_at": _created_at(),
        "submission_status": evidence["submission_status"],
        "primary_claim": claim["claim"],
        "claim_metadata": claim["metadata"],
        "claim_boundaries": claim["boundaries"],
        "reviewer_risks": {
            "rows": risks["rows"],
            "resolved": risks["resolved"],
        },
        "gates": evidence["gates"],
        "artifact_count": len(artifacts),
        "file_count": len(files),
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "files": files,
        "provenance": {
            "report": "outputs/reports/cvpr_artifact_provenance_v2.json",
            "generated_artifacts": provenance["generated_artifact_count"],
            "mapped_artifacts": provenance["mapped_artifact_count"],
        },
        "verification": {
            "pytest": _pytest(pytest_report),
            "ruff": ruff_report.read_text(encoding="utf-8").strip(),
            "python_line_limit": _read("outputs/reports/fp_line_limit_v18.json"),
        },
        "archive_policy": (
            "Sorted paths; fixed mtime, owner, group, uid, and gid; normalized "
            "modes; gzip mtime=0."
        ),
    }
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for path in artifacts:
        target = PACKAGE / "artifacts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for relative in data_directories():
        shutil.copytree(ROOT / relative, PACKAGE / "data" / relative)
    (PACKAGE / "README.md").write_bytes(readme_bytes)
    (PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    actual = sorted(
        f"{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}"
        for path in PACKAGE.rglob("*")
        if path.is_file()
    )
    expected = [row["path"] for row in files]
    if actual != expected:
        raise RuntimeError("package files differ from manifest")
    archive_hash = build_deterministic_tar_gz(PACKAGE, ARCHIVE)
    with tempfile.TemporaryDirectory(prefix="fevp_v18_builder_") as temp:
        rebuilt = Path(temp) / ARCHIVE.name
        rebuilt_hash = build_deterministic_tar_gz(PACKAGE, rebuilt)
    if rebuilt_hash != archive_hash:
        raise RuntimeError("deterministic archive rebuild mismatch")
    ARCHIVE.with_name(f"{ARCHIVE.name}.sha256").write_text(
        f"{archive_hash}  {ARCHIVE.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifacts": len(artifacts),
                "files": len(files),
                "archive_sha256": archive_hash,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
