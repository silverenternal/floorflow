#!/usr/bin/env python3
"""Assemble the repaired v45 15-design layout cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrity",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_external_pair_integrity.json",
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_collapse_sensitivity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_repaired_cohort_manifest.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_row(
    case: str,
    source: str,
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    hashes = [candidate["tcl_sha256"], control["tcl_sha256"]]
    routes = [candidate["legalizer_route"], control["legalizer_route"]]
    return {
        "case": case,
        "pair_source": source,
        "candidate": candidate,
        "control": control,
        "distinct_legalized_placements": bool(
            all(hashes) and hashes[0] != hashes[1]
        ),
        "common_route": routes[0] == routes[1],
        "routed_endpoint_status": (
            "complete_original" if source == "original_valid" else "pending_sensitivity"
        ),
    }


def main() -> None:
    args = parse_args()
    integrity = load(args.integrity)
    sensitivity = load(args.sensitivity)
    audit_cases = {row["case"]: row for row in integrity["cases"]}
    sensitivity_cases = {row["case"]: row for row in sensitivity["cases"]}
    valid = [pair_row(case, "original_valid", audit_cases[case]["candidate"], audit_cases[case]["control"]) for case in integrity["valid_completed_cases"]]
    corrected = [
        pair_row(
            case,
            "corrected_sensitivity",
            sensitivity_cases[case]["arms"]["candidate"],
            sensitivity_cases[case]["arms"]["control"],
        )
        for case in integrity["collapsed_cases"]
    ]
    pairs = valid + corrected
    cases = [row["case"] for row in pairs]
    checks = {
        "case_count_is_15": len(pairs) == 15,
        "cases_are_unique": len(cases) == len(set(cases)),
        "all_pairs_distinct": all(row["distinct_legalized_placements"] for row in pairs),
        "all_pairs_common_route": all(row["common_route"] for row in pairs),
        "valid_and_corrected_disjoint": not (
            set(integrity["valid_completed_cases"])
            & set(integrity["collapsed_cases"])
        ),
    }
    report = {
        "version": "chip-v45-repaired-cohort-v1",
        "status": "layout_ready_routed_pending" if all(checks.values()) else "reject",
        "scope": (
            "Eight original valid pairs plus seven corrected sensitivity pairs. "
            "This is an observed repaired cohort, not an untouched confirmation set."
        ),
        "counts": {
            "original_valid": len(valid),
            "corrected_sensitivity": len(corrected),
            "total": len(pairs),
            "routed_complete": len(valid),
            "routed_pending": len(corrected),
        },
        "checks": checks,
        "pairs": pairs,
        "claim_boundary": (
            "No 15-design routed-PPA claim is allowed until all seven corrected "
            "pairs finish and are certified. This cohort can never replace the "
            "fresh untouched holdout."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
