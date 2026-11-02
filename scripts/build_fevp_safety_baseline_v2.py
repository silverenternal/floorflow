#!/usr/bin/env python3
"""Build bounded safety-baseline claims for validity and watertightness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validity",
        type=Path,
        default=ROOT / "outputs/reports/fevp_validity_watertight_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_safety_baseline_v2.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs/fevp_safety_baseline_v2.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.validity.read_text(encoding="utf-8"))
    summary = source["summary"]["all"]
    output = {
        "status": "complete",
        "claim_role": "safety_baseline_not_primary_method_gain",
        "validity": summary["validity"],
        "watertight_before": summary["before_watertight_rate"],
        "watertight_after": summary["after_watertight_rate"],
        "watertight_diagnostics": summary["watertight"],
        "raw_diagnostics": summary["raw_watertight"],
        "repair_count": summary["repair_count"],
        "regression_count": summary["regression_count"],
        "repair_rate_of_failures": summary["repair_rate_of_failures"],
        "allowed_claims": [
            "Validity is reported as a five-check conjunction.",
            "Watertight failures are localized with hole and dangling-edge diagnostics.",
            "The guard's repair and regression counts are both disclosed.",
        ],
        "forbidden_claims": [
            "Validity is 1.0 under the hardened five-check definition.",
            "The watertight guard provides a net improvement.",
            "Watertightness is a primary contribution of the method.",
            "The guard should be promoted based on repaired cases alone.",
        ],
        "claim_text": (
            f"The five-check validity is {summary['validity']['validity']:.4f}, "
            "and watertightness is used as an auditable safety baseline. The "
            f"guard changes conditional watertightness from "
            f"{summary['before_watertight_rate']:.4f} to "
            f"{summary['after_watertight_rate']:.4f}, repairing "
            f"{summary['repair_count']} failures while introducing "
            f"{summary['regression_count']} regressions; it is not claimed as a "
            "net method improvement."
        ),
        "checks": {
            "validity_not_reported_as_one": (
                summary["validity"]["validity"] < 1.0
            ),
            "guard_not_promoted": (
                summary["repair_count"] <= summary["regression_count"]
            ),
            "five_checks_present": all(
                key in summary["validity"]
                for key in (
                    "outer_boundary_closed",
                    "room_polygon_closed",
                    "no_self_intersection",
                    "no_room_overlap",
                    "semantic_complete",
                )
            ),
        },
        "source": str(args.validity.relative_to(ROOT)),
    }
    failures = [key for key, value in output["checks"].items() if not value]
    if failures:
        output["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    doc = [
        "# Validity and Watertightness Safety Boundary",
        "",
        output["claim_text"],
        "",
        "## Allowed",
        "",
        *[f"- {claim}" for claim in output["allowed_claims"]],
        "",
        "## Forbidden",
        "",
        *[f"- {claim}" for claim in output["forbidden_claims"]],
        "",
    ]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(doc), encoding="utf-8")
    print(json.dumps({"status": output["status"], **output["checks"]}, indent=2))
    raise SystemExit(0 if output["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
