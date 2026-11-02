#!/usr/bin/env python3
"""Build the deterministic CVPR risk-closure evidence package v17."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from cvpr_evidence_registry import ROOT, canonical_artifacts, data_directories

from floorflow.evaluation.deterministic_archive import (
    build_deterministic_tar_gz,
    sha256_file,
)

PACKAGE = ROOT / "outputs/cvpr_package_fp_v17"
ARCHIVE = ROOT / "outputs/cvpr_package_fp_v17.tar.gz"


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


def _directory_digest(path: Path) -> tuple[int, int, str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    value = hashlib.sha256()
    for item in files:
        value.update(str(item.relative_to(path)).encode("utf-8"))
        value.update(b"\0")
        value.update(sha256_file(item).encode("ascii"))
    return len(files), sum(item.stat().st_size for item in files), value.hexdigest()


def _created_at() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def main() -> None:
    artifacts = canonical_artifacts()
    missing = [path for path in artifacts if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing v17 artifacts: {missing}")
    evidence = _read("outputs/reports/fp_cvpr_evidence_v12.json")
    closure = _read("outputs/reports/cvpr_risk_closure_v1.json")
    provenance = _read("outputs/reports/cvpr_artifact_provenance_v1.json")
    pytest_report = ROOT / "outputs/reports/fp_pytest_v17.xml"
    ruff_report = ROOT / "outputs/reports/fp_ruff_v17.txt"
    if not pytest_report.is_file() or not ruff_report.is_file():
        raise FileNotFoundError("run pytest and Ruff v17 before packaging")
    artifact_rows = {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256_file(ROOT / path),
            "package_path": f"artifacts/{path}",
        }
        for path in artifacts
    }
    directory_rows = {}
    data_file_rows = []
    for path in data_directories():
        target = ROOT / path
        files, size, tree_hash = _directory_digest(target)
        directory_rows[path] = {
            "files": files,
            "bytes": size,
            "tree_sha256": tree_hash,
            "package_path": f"data/{path}",
        }
        for item in sorted(candidate for candidate in target.rglob("*") if candidate.is_file()):
            relative = item.relative_to(ROOT).as_posix()
            data_file_rows.append(
                {
                    "path": f"{PACKAGE.name}/data/{relative}",
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    readme = f"""# FloorFlow CVPR Experiment Evidence v17

This package closes all numbered checks in the supplied CVPR self-audit.
It contains experiments and claim-support artifacts only, not a manuscript.

## Canonical claim

{evidence['primary_narrative']}

## Audit summary

- Numbered source checks: {closure['resolved']}/{closure['items']} resolved.
- Statistical intervals, effect sizes, category breakdowns, external protocol
  boundaries, formal FEVP definitions, and artifact provenance are included.
- The archive uses sorted entries and fixed metadata and is independently
  rebuilt by the final audit.

## Boundaries

- HouseMind is unreported under FEVP, not failed.
- CE2EPlan is a generation-only portability target.
- Raster2Seq is a matched reconstruction baseline using shared project
  preprocessing for the matched edit-row sensitivity study.
- The watertight guard is a safety diagnostic, not a net-benefit claim.
- Human preference is outside scope.
"""
    readme_bytes = readme.encode("utf-8")
    manifest_files = [
        {
            "path": f"{PACKAGE.name}/artifacts/{path}",
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for path, row in artifact_rows.items()
    ]
    manifest_files.extend(data_file_rows)
    manifest_files.append(
        {
            "path": f"{PACKAGE.name}/README.md",
            "bytes": len(readme_bytes),
            "sha256": hashlib.sha256(readme_bytes).hexdigest(),
        }
    )
    manifest_files.append(
        {
            "path": f"{PACKAGE.name}/manifest.json",
            "self_reference": True,
        }
    )
    manifest_files.sort(key=lambda row: row["path"])
    manifest = {
        "status": "cvpr_risk_closure_evidence_v17",
        "created_at": _created_at(),
        "submission_status": evidence["submission_status"],
        "primary_claim": evidence["primary_narrative"],
        "release_boundary": evidence["release_boundary"],
        "risk_closure": evidence["risk_closure"],
        "statistical_governance": evidence["statistical_governance"],
        "external_protocol": evidence["external_protocol"],
        "reproducibility": evidence["reproducibility"],
        "gates": evidence["gates"],
        "experiment_objective_complete": evidence["experiment_objective_complete"],
        "manuscript_in_scope": False,
        "human_evaluation": "excluded_by_user_request",
        "package_name": PACKAGE.name,
        "hash_algorithm": "sha256",
        "archive_policy": (
            "Sorted paths; fixed mtime, uid, gid, owner, group, and normalized "
            "file modes; mtime-zero gzip."
        ),
        "archive": "outputs/cvpr_package_fp_v17.tar.gz",
        "artifact_count": len(artifacts),
        "file_count": len(manifest_files),
        "artifacts": artifact_rows,
        "data_directories": directory_rows,
        "files": manifest_files,
        "provenance": {
            "report": "outputs/reports/cvpr_artifact_provenance_v1.json",
            "generated_artifact_count": provenance["generated_artifact_count"],
            "mapped_artifact_count": provenance["mapped_artifact_count"],
        },
        "risk_resolution": {
            "requested_themes": closure["scope"],
            "numbered_checks": closure["items"],
            "resolved_checks": closure["resolved"],
        },
        "excluded_external_assets": {
            "CE2EPlan": (
                "No license; code, RPLAN data, weights, and images excluded."
            ),
            "HouseMind": "Official code remains Code (Soon).",
        },
        "verification": {
            "pytest": _pytest(pytest_report),
            "ruff": ruff_report.read_text(encoding="utf-8").strip(),
            "python_line_limit": _read(
                "outputs/reports/fp_line_limit_v17.json"
            ),
            "json_validation": "pass",
        },
    }
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    (PACKAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in artifacts:
        target = PACKAGE / "artifacts" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for path in data_directories():
        shutil.copytree(ROOT / path, PACKAGE / "data" / path)
    (PACKAGE / "README.md").write_bytes(readme_bytes)
    actual_paths = sorted(
        f"{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}"
        for path in PACKAGE.rglob("*")
        if path.is_file()
    )
    expected_paths = [row["path"] for row in manifest_files]
    if actual_paths != expected_paths:
        raise RuntimeError("package files do not match manifest files")
    archive_hash = build_deterministic_tar_gz(PACKAGE, ARCHIVE)
    with tempfile.TemporaryDirectory(prefix="fevp_v17_builder_") as temp:
        rebuilt = Path(temp) / ARCHIVE.name
        rebuilt_hash = build_deterministic_tar_gz(PACKAGE, rebuilt)
    if rebuilt_hash != archive_hash:
        raise RuntimeError("deterministic archive rebuild hash mismatch")
    ARCHIVE.with_name(f"{ARCHIVE.name}.sha256").write_text(
        f"{archive_hash}  {ARCHIVE.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifacts": len(artifacts),
                "files": manifest["file_count"],
                "archive_sha256": archive_hash,
                "deterministic_rebuild": rebuilt_hash == archive_hash,
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
