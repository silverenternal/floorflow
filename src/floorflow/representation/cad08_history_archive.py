"""Opt-in external transition-history archives, separate from CAD and scoring.

Hashes identify bytes supplied by the caller, not authorship, permissions, or
cross-rebuild object identity. Archives do not embed CAD or native snapshots.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from floorflow.representation.cad08_transition_history import verify_history

ARCHIVE_SCHEMA = "floorflow-cad08-history-archive-v1"


def make_archive(branch: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Package copies of transition records; do not claim they are validated."""
    return {"schema": ARCHIVE_SCHEMA, "branch": branch, "records": copy.deepcopy(records)}


def append_transition(
    archive: dict[str, Any], record: dict[str, Any], *, trusted_current_head: str
) -> dict[str, Any]:
    """Return a new archive linked to a caller-supplied current chain head.

    This checks only the explicit append boundary. Call ``verify_archive`` with
    independently supplied snapshots and endpoints to assess all local links.
    """
    records = archive["records"]
    if archive.get("schema") != ARCHIVE_SCHEMA or not records:
        raise ValueError("An existing versioned, non-empty archive is required")
    if records[-1].get("record_hash") != trusted_current_head:
        raise ValueError("The archive head differs from the trusted current head")
    if (record.get("previous_head") != trusted_current_head or record.get("sequence") != len(records)
            or record.get("branch") != archive.get("branch")):
        raise ValueError("Transition does not continue the selected archive")
    return make_archive(archive["branch"], [*records, record])


def write_archive(path: Path, archive: dict[str, Any]) -> str:
    """Write a new artifact without overwriting; return its exact byte SHA-256."""
    data = (json.dumps(archive, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def read_archive(path: Path, *, trusted_file_sha256: str | None) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """Read against an external file hash; missing trust is unknown, mismatch fails."""
    data = path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if trusted_file_sha256 is not None and observed != trusted_file_sha256:
        return None, {"file_integrity": "fail", "observed_sha256": observed}
    try:
        archive = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None, {"file_integrity": "fail", "observed_sha256": observed}
    if not isinstance(archive, dict) or archive.get("schema") != ARCHIVE_SCHEMA:
        return None, {"file_integrity": "fail", "observed_sha256": observed}
    return archive, {"file_integrity": "pass" if trusted_file_sha256 is not None else "unknown",
                     "observed_sha256": observed}


def verify_archive(
    archive: dict[str, Any], snapshots: list[dict[str, Any] | None], *, branch: str,
    trusted_previous_head: str | None, trusted_head: str | None,
    action_evidence: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Verify record links and supplied snapshots; do not assess historical safety."""
    if archive.get("schema") != ARCHIVE_SCHEMA or archive.get("branch") != branch:
        return {"status": "fail", "link_integrity": "fail", "reason": "archive_schema_or_branch_mismatch"}
    return verify_history(archive["records"], snapshots, branch=branch,
                          trusted_previous_head=trusted_previous_head, trusted_head=trusted_head,
                          action_evidence=action_evidence)
