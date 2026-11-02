"""Build and validate a replayable contour-only evaluation manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contour_only_report(path: str | Path) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("masked_control", False):
        raise ValueError(f"{path} uses masked target-derived control")
    if report.get("guided_control", False):
        raise ValueError(f"{path} uses guided target-derived control")
    provenance = report.get("condition_provenance")
    if isinstance(provenance, dict):
        for mode, row in provenance.items():
            if isinstance(row, dict) and row.get("mode") != "contour_only":
                raise ValueError(f"{path} contains non-contour provenance {mode}")
    return report


def build_manifest(
    *,
    split: str,
    checkpoints: dict[str, str],
    artifacts: dict[str, str],
    command: list[str],
    cache_root: str,
) -> dict:
    reports = {}
    for name, path in artifacts.items():
        if Path(path).suffix.lower() != ".json":
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            reports[name] = validate_contour_only_report(path)
    return {
        "version": "contour-only-v1",
        "condition_mode": "contour_only",
        "target_control_free": True,
        "best_of_n_used": False,
        "split": split,
        "command": list(command),
        "cache_root": cache_root,
        "checkpoints": {
            name: {"path": path, "sha256": sha256_file(path)}
            for name, path in checkpoints.items()
        },
        "artifacts": {
            name: {"path": path, "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "report_cases": {
            name: report.get("cases", report.get("case_count"))
            for name, report in reports.items()
        },
    }


__all__ = [
    "build_manifest",
    "sha256_file",
    "validate_contour_only_report",
]
