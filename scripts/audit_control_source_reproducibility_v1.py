#!/usr/bin/env python3
"""Compare NPZ SHA-256 values across repeated control-source builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/control_source_reproducibility_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    reference_rows = {row["name"]: row for row in reference["rows"]}
    candidate_rows = {row["name"]: row for row in candidate["rows"]}
    missing = sorted(set(candidate_rows) - set(reference_rows))
    matches = [
        candidate_rows[name]["sha256"] == reference_rows[name]["sha256"]
        for name in sorted(set(candidate_rows) & set(reference_rows))
    ]
    report = {
        "version": "control-source-reproducibility-v1",
        "reference_manifest": str(args.reference_manifest),
        "candidate_manifest": str(args.candidate_manifest),
        "reference_cases": len(reference_rows),
        "candidate_cases": len(candidate_rows),
        "missing_from_reference": missing,
        "compared_cases": len(matches),
        "exact_sha256_matches": sum(matches),
        "all_exact": not missing and bool(matches) and all(matches),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
