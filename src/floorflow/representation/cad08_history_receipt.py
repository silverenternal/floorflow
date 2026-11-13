"""File-identity receipt for CAD export artifacts and external history.

This module does not parse DXF or establish geometric or semantic equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from floorflow.representation.cad08_history_archive import ARCHIVE_SCHEMA, read_archive, verify_archive

RECEIPT_SCHEMA = "floorflow-cad08-history-receipt-v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_history_receipt(
    path: Path, *, dxf: Path, metadata: Path, history: Path,
    before_export_cad_content_sha256: str, branch: str, head: str, stage: str,
) -> str:
    """Write a new external receipt and return its byte hash to the caller.

    This reads existing artifacts only; caller must independently establish
    that the supplied CAD digest, branch and head correspond to the export.
    """
    record = {
        "schema": RECEIPT_SCHEMA, "history_schema": ARCHIVE_SCHEMA,
        "files": {"dxf": _sha(dxf), "metadata": _sha(metadata), "history": _sha(history)},
        "before_export_cad_content_sha256": before_export_cad_content_sha256,
        "branch": branch, "head": head, "stage": stage,
    }
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def verify_history_receipt(
    receipt_path: Path, *, dxf: Path, metadata: Path, history: Path,
    trusted_receipt_sha256: str | None, branch: str, trusted_previous_head: str | None,
    trusted_head: str | None, snapshots: list[dict[str, Any] | None],
    action_evidence: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Check file bytes and supplied history independently; never interpret DXF.

    Trusted receipt and chain endpoints must come from independent caller-held
    sources. A file-hash pass is not a geometry or safety result.
    """
    try:
        raw = receipt_path.read_bytes()
    except FileNotFoundError:
        return {"receipt_integrity": "unknown", "file_integrity": "unknown", "history_integrity": "unknown",
                "declared_terminal_content_match": "unknown", "geometry": "not_evaluated",
                "original_epgj": "not_evaluated", "permissions": "unknown", "cross_rebuild_identity": "unknown",
                "reason": "receipt_missing"}
    observed = hashlib.sha256(raw).hexdigest()
    base: dict[str, Any] = {
        "receipt_integrity": "unknown", "file_integrity": "unknown", "history_integrity": "unknown",
        "declared_terminal_content_match": "unknown", "geometry": "not_evaluated", "original_epgj": "not_evaluated",
        "permissions": "unknown", "cross_rebuild_identity": "unknown", "observed_receipt_sha256": observed,
    }
    if trusted_receipt_sha256 is not None and observed != trusted_receipt_sha256:
        return {**base, "receipt_integrity": "fail"}
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {**base, "receipt_integrity": "fail"}
    if not isinstance(record, dict) or record.get("schema") != RECEIPT_SCHEMA:
        return {**base, "receipt_integrity": "fail"}
    base["receipt_integrity"] = "pass" if trusted_receipt_sha256 is not None else "unknown"
    if base["receipt_integrity"] != "pass":
        # Do not promote hashes from the same unauthenticated receipt into trust.
        return base
    if record.get("history_schema") != ARCHIVE_SCHEMA or record.get("branch") != branch:
        return {**base, "file_integrity": "fail"}
    files = record.get("files")
    if not isinstance(files, dict):
        return {**base, "file_integrity": "fail"}
    paths = (("dxf", dxf), ("metadata", metadata), ("history", history))
    missing = [key for key, path in paths if not path.is_file()]
    if missing:
        return {**base, "missing_files": missing}
    if any(files.get(key) != _sha(path) for key, path in paths):
        return {**base, "file_integrity": "fail"}
    base["file_integrity"] = "pass"
    if trusted_head is not None and record.get("head") != trusted_head:
        return {**base, "history_integrity": "fail"}
    archive, archive_result = read_archive(history, trusted_file_sha256=files["history"])
    if archive_result["file_integrity"] != "pass" or archive is None:
        return {**base, "history_integrity": "fail"}
    verification = verify_archive(archive, snapshots, branch=branch, trusted_previous_head=trusted_previous_head,
                                  trusted_head=trusted_head, action_evidence=action_evidence)
    base["history_integrity"] = verification["link_integrity"]
    base["history_verification"] = verification
    if archive.get("records"):
        final_cad = archive["records"][-1].get("after_cad")
        claimed = record.get("before_export_cad_content_sha256")
        if final_cad is not None and claimed is not None:
            base["declared_terminal_content_match"] = "pass" if final_cad == claimed else "fail"
    return base
