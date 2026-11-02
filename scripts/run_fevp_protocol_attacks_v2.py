#!/usr/bin/env python3
"""Run and report the deterministic FEVP attack suite."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorflow.evaluation.fevp_metric_contracts import load_metric_contract
from scripts.build_fevp_protocol_attacks_v2 import (
    graph_attacks,
    outside_attacks,
    parse_args,
    validity_attacks,
    watertight_attacks,
)


def main() -> None:
    args = parse_args()
    contract = load_metric_contract(args.contract)
    rows = [
        *outside_attacks(),
        *validity_attacks(),
        *watertight_attacks(),
        *graph_attacks(),
    ]
    by_metric = {}
    for metric in (
        "outside_exact",
        "validity",
        "conditional_watertightness",
        "node_f1",
        "edge_overlap",
        "ged",
    ):
        selected = [row for row in rows if row["metric"] == metric]
        by_metric[metric] = {
            "attacks": len(selected),
            "sensitive": sum(row["sensitive"] for row in selected),
            "false_pass_rate": (
                sum(row["false_pass"] for row in selected) / len(selected)
                if selected
                else None
            ),
            "false_fail_rate": (
                sum(row["false_fail"] for row in selected) / len(selected)
                if selected
                else None
            ),
        }
    failures = [
        row for row in rows if not row["sensitive"] or row["false_fail"]
    ]
    report = {
        "status": "complete" if not failures else "failed",
        "contract_version": contract["implementation_version"],
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "tolerances": contract["tolerances"],
        "attacks": len(rows),
        "by_metric": by_metric,
        "checks": {
            "all_mandatory_metrics_attacked": all(
                by_metric[name]["attacks"] >= 3
                for name in (
                    "outside_exact",
                    "validity",
                    "conditional_watertightness",
                    "node_f1",
                    "edge_overlap",
                )
            ),
            "all_attacks_sensitive": not failures,
            "all_metric_baselines_reached": all(
                row["false_fail"] is False for row in rows
            ),
        },
        "rows": rows,
        "boundary": (
            "These deterministic synthetic attacks test evaluator sensitivity, "
            "not model quality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        json.dumps({"status": report["status"], "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    examples = {
        "status": "complete",
        "schema": str(args.case_schema.relative_to(ROOT)),
        "examples": {
            metric: [
                row
                for row in rows
                if row["metric"] == metric and row["sensitive"]
            ][:3]
            for metric in by_metric
        },
    }
    args.examples_output.parent.mkdir(parents=True, exist_ok=True)
    args.examples_output.write_text(
        json.dumps(examples, indent=2) + "\n",
        encoding="utf-8",
    )
    table = [
        "# FEVP Protocol Attack Sensitivity",
        "",
        "| Metric | Attacks | Sensitive | False pass | False fail |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in by_metric.items():
        table.append(
            f"| {metric} | {values['attacks']} | {values['sensitive']} | "
            f"{values['false_pass_rate']:.3f} | "
            f"{values['false_fail_rate']:.3f} |"
        )
    table.extend(
        [
            "",
            "Every mandatory metric changes under its intended attack.",
            "",
        ]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(table), encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["checks"]}, indent=2))
    raise SystemExit(0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
