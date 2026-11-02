"""Verify the residential evidence archive against its manifest and checksums."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "residential_sota_evidence_v4"
PACKAGE_ROOT = ROOT / "outputs" / PACKAGE_NAME
ARCHIVE = ROOT / "outputs" / f"{PACKAGE_NAME}.tar.gz"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    archive_before = sha256(ARCHIVE)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_residential_sota_evidence_v4.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(
        (PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    declared = {row["path"] for row in manifest["files"]}
    with tarfile.open(ARCHIVE, mode="r:gz") as archive:
        archived = {
            member.name
            for member in archive.getmembers()
            if member.isfile()
        }
    expected_archived = declared | {f"{PACKAGE_NAME}/manifest.json"}
    actual_hashes = {
        row["path"]: sha256(ROOT / "outputs" / row["path"])
        for row in manifest["files"]
    }
    checksum_lines = (
        PACKAGE_ROOT / "SHA256SUMS"
    ).read_text(encoding="utf-8").splitlines()
    report = {
        "status": "passed",
        "scope": "residential-only evidence package integrity",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "declared_files": len(declared),
        "archived_files": len(archived),
        "checksum_entries": len(checksum_lines),
        "checks": {
            "manifest_matches_archive": archived == expected_archived,
            "all_declared_hashes_match": all(
                actual_hashes[path] == row["sha256"]
                for row, path in zip(
                    manifest["files"],
                    actual_hashes,
                    strict=True,
                )
            ),
            "no_chips_paths": not any("chip" in path.lower() for path in archived),
            "deterministic_rebuild": archive_before == sha256(ARCHIVE),
        },
    }
    report["status"] = (
        "passed" if all(report["checks"].values()) else "not_passed"
    )
    output = ROOT / "outputs/reports/residential_sota_package_audit_v4.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
