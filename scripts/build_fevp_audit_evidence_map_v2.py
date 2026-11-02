#!/usr/bin/env python3
"""Build the editing-task auditability evidence map."""

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
        default=ROOT / "outputs/reports/fevp_audit_evidence_map_v2.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper/tables/fevp_audit_evidence_map_v2.md",
    )
    return parser.parse_args()


def _row(
    method: str,
    role: str,
    outside_exact: str,
    violation: str,
    negative: str,
    attacks: str,
    paired: str,
) -> dict:
    return {
        "method": method,
        "role": role,
        "outside_exact": outside_exact,
        "violation_rate": violation,
        "negative_controls": negative,
        "adversarial_checks": attacks,
        "paired_statistics": paired,
    }


def main() -> None:
    args = parse_args()
    rows = [
        _row(
            "FEVP + FloorFlow",
            "evaluated",
            "✓",
            "✓",
            "✓",
            "✓",
            "✓",
        ),
        _row(
            "HouseMind-E",
            "editing context",
            "—",
            "—",
            "—",
            "—",
            "—",
        ),
        _row(
            "HouseMind-O",
            "unified editing context",
            "—",
            "—",
            "—",
            "—",
            "—",
        ),
        _row(
            "HouseMind-G",
            "generation only",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
        ),
        _row(
            "CE2EPlan",
            "generation context",
            "—",
            "—",
            "—",
            "—",
            "—",
        ),
        _row(
            "Raster2Seq",
            "reconstruction context",
            "—",
            "—",
            "—",
            "—",
            "—",
        ),
        _row(
            "Raster2Seq outside-frozen wrapper",
            "editing wrapper",
            "pixel only",
            "—",
            "—",
            "—",
            "—",
        ),
    ]
    report = {
        "status": "complete",
        "columns": {
            "outside_exact": "Strict outside pixel/room persistence under a matched edit protocol.",
            "violation_rate": "Continuous outside changed-area rate.",
            "negative_controls": "Single-room or deliberately untouched-room controls.",
            "adversarial_checks": "Small corruptions that verify evaluator sensitivity.",
            "paired_statistics": "Paired CIs/tests over matched cases or seeds.",
        },
        "rows": rows,
        "footnote": (
            "— means the method does not report the corresponding audit evidence; "
            "it does not mean the method lacks the property or performs worse."
        ),
        "checks": {
            "floorflow_complete": all(
                rows[0][key] == "✓"
                for key in (
                    "outside_exact",
                    "violation_rate",
                    "negative_controls",
                    "adversarial_checks",
                    "paired_statistics",
                )
            ),
            "housemind_not_ranked": all(
                row["outside_exact"] == "—"
                for row in rows
                if row["method"].startswith("HouseMind") and row["method"] != "HouseMind-G"
            ),
            "generation_only_marked": next(
                row for row in rows if row["method"] == "HouseMind-G"
            )["role"]
            == "generation only",
        },
    }
    failures = [key for key, value in report["checks"].items() if not value]
    if failures:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = [
        "# Editing-task Audit Evidence Map",
        "",
        "| Method | Role | Outside exact | Violation rate | Negative controls | Adversarial checks | Paired stats |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['role']} | {row['outside_exact']} | "
            f"{row['violation_rate']} | {row['negative_controls']} | "
            f"{row['adversarial_checks']} | {row['paired_statistics']} |"
        )
    table.extend(["", report["footnote"], ""])
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
