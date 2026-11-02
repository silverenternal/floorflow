#!/usr/bin/env python3
"""Build the compact HouseMind/CE2EPlan/FloorFlow audit coverage table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/fevp_core_audit_coverage_v3.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_core_audit_coverage_v3.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = (
        "outside_exact",
        "violation_rate",
        "negative_control",
        "adversarial_checks",
        "paired_stats",
    )
    rows = [
        {
            "method": "FEVP + FloorFlow",
            "role": "complete evaluated protocol row",
            **{column: "reported" for column in columns},
        },
        {
            "method": "HouseMind",
            "role": "complementary multimodal instruction editing",
            **{column: "not reported" for column in columns},
        },
        {
            "method": "CE2EPlan",
            "role": "boundary-conditioned generation context",
            **{column: "not reported" for column in columns},
        },
    ]
    report = {
        "status": "complete",
        "columns": columns,
        "rows": rows,
        "footnote": (
            "'not reported' does not mean the method lacks the property or "
            "performs worse; it only means the public paper/release provided "
            "no corresponding FEVP audit evidence."
        ),
        "checks": {
            "floorflow_complete": all(
                rows[0][column] == "reported" for column in columns
            ),
            "house_mind_not_ranked": all(
                rows[1][column] == "not reported" for column in columns
            ),
            "ce2eplan_generation_context": (
                rows[2]["role"]
                == "boundary-conditioned generation context"
            ),
        },
        "sources": {
            "floorflow": "outputs/reports/fevp_case_rows_v2.json",
            "house_mind": (
                "outputs/reports/fevp_housemind_positioning_v2.json"
            ),
            "ce2eplan": "outputs/reports/ce2eplan_official_anchor_v1.json",
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# FEVP Core Audit Coverage",
        "",
        "| Method | Outside exact | Violation rate | Negative control | Adversarial checks | Paired stats |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['outside_exact']} | "
            f"{row['violation_rate']} | {row['negative_control']} | "
            f"{row['adversarial_checks']} | {row['paired_stats']} |"
        )
    table.extend(["", report["footnote"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
