#!/usr/bin/env python3
"""Audit FEVP-2 artifact provenance against current source files."""

from __future__ import annotations

import json
from pathlib import Path

from cvpr_vector_artifact_registry import vector_canonical_artifacts

from floorflow.evaluation.deterministic_archive import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "outputs/reports/cvpr_artifact_provenance_v2.json"
OUTPUT = ROOT / "outputs/reports/cvpr_artifact_provenance_audit_v2.json"


def _generated(artifacts: list[str]) -> set[str]:
    return {
        path
        for path in artifacts
        if (
            path.startswith("outputs/reports/")
            and path.endswith(".json")
        )
        or path.startswith("paper/tables/")
        and path.endswith(".md")
    }


def main() -> None:
    report = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    expected = _generated(vector_canonical_artifacts())
    mapped = set(report["artifacts"])
    errors = []
    if mapped != expected:
        errors.extend(f"missing provenance: {path}" for path in sorted(expected - mapped))
        errors.extend(f"unexpected provenance: {path}" for path in sorted(mapped - expected))
    checked_inputs = 0
    for target, row in report["artifacts"].items():
        if not (ROOT / target).is_file():
            errors.append(f"missing artifact: {target}")
        generator = ROOT / row["generator"]
        if not generator.is_file():
            errors.append(f"missing generator: {row['generator']}")
            continue
        if sha256_file(generator) != row["generator_sha256"]:
            errors.append(f"stale generator: {row['generator']}")
        for input_path, digest in row["inputs"].items():
            checked_inputs += 1
            path = ROOT / input_path
            if not path.is_file() or sha256_file(path) != digest:
                errors.append(f"stale input: {input_path}")
    audit = {
        "status": "complete" if not errors else "failed",
        "generated_artifacts": len(expected),
        "mapped_artifacts": len(mapped),
        "checked_inputs": checked_inputs,
        "errors": errors,
        "checks": {
            "all_generated_artifacts_mapped": mapped == expected,
            "all_generators_current": not any(
                "stale generator" in error for error in errors
            ),
            "all_inputs_current": not any(
                "stale input" in error for error in errors
            ),
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
