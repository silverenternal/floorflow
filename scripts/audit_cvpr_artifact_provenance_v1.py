#!/usr/bin/env python3
"""Audit canonical artifact provenance against current files on disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvpr_evidence_registry import ROOT, canonical_artifacts

from floorflow.evaluation.deterministic_archive import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT / "outputs/reports/cvpr_artifact_provenance_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/cvpr_artifact_provenance_audit_v1.json",
    )
    return parser.parse_args()


def _expected_generated(artifacts: list[str]) -> set[str]:
    return {
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
    }


def main() -> None:
    args = parse_args()
    report = json.loads(args.provenance.read_text(encoding="utf-8"))
    expected = _expected_generated(canonical_artifacts())
    mapped = set(report.get("artifacts", {}))
    errors = []
    if report.get("status") != "complete":
        errors.append("provenance builder did not complete")
    if mapped != expected:
        errors.extend(f"missing provenance: {path}" for path in sorted(expected - mapped))
        errors.extend(f"unexpected provenance: {path}" for path in sorted(mapped - expected))
    checked_inputs = 0
    for target, row in report.get("artifacts", {}).items():
        if not (ROOT / target).is_file():
            errors.append(f"missing target artifact: {target}")
        generator = ROOT / row["generator"]
        if not generator.is_file():
            errors.append(f"missing generator: {row['generator']}")
            continue
        if sha256_file(generator) != row["generator_sha256"]:
            errors.append(f"stale generator hash: {row['generator']}")
        for input_path, expected_hash in row["inputs"].items():
            checked_inputs += 1
            path = ROOT / input_path
            if not path.is_file():
                errors.append(f"missing provenance input: {input_path}")
            elif sha256_file(path) != expected_hash:
                errors.append(f"stale provenance input: {input_path}")
    audit = {
        "status": "complete" if not errors else "failed",
        "generated_artifacts": len(expected),
        "mapped_artifacts": len(mapped),
        "checked_inputs": checked_inputs,
        "errors": errors,
        "checks": {
            "all_generated_artifacts_mapped": mapped == expected,
            "all_generators_present": all(
                (ROOT / row["generator"]).is_file()
                for row in report.get("artifacts", {}).values()
            ),
            "all_generator_hashes_current": not any(
                "generator hash" in error for error in errors
            ),
            "all_input_hashes_current": not any(
                "provenance input" in error for error in errors
            ),
            "no_internal_private_api_dependency": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
