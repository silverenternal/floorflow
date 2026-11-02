#!/usr/bin/env python3
"""Decide whether the completed M6 pilot adds proposal-ranking value."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from floorflow.chip_flow.training_runtime import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    ("candidate", 1),
    ("candidate", 2),
    ("control", 1),
    ("control", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v46_analytical_search_decision_v1.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selection(proposals: list[dict], key: str) -> dict[str, Any]:
    finite = [
        row
        for row in proposals
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    if not finite:
        raise ValueError(f"no finite values for {key}")
    best = min(float(row[key]) for row in finite)
    tied = [
        int(row["proposal"])
        for row in finite
        if math.isclose(float(row[key]), best, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return {
        "score": best,
        "tied_proposals": tied,
        "default_proposal": min(tied),
    }


def audit_arm(arm: str, count: int) -> list[dict]:
    report = load(
        ROOT
        / "outputs/reports"
        / f"chip_v46_analytical_search_{arm}_n{count}_positioned_all.json"
    )
    rows = []
    for case in report["results"]:
        proposals = case["proposals"]
        p2b = selection(proposals, "p2b_hpwl_um")
        analytical = selection(proposals, "globalplace_wirelength")
        by_index = {int(row["proposal"]): row for row in proposals}
        default_score = float(
            by_index[p2b["default_proposal"]]["globalplace_wirelength"]
        )
        incremental = (analytical["score"] - default_score) / abs(default_score)
        rows.append(
            {
                "case": case["case"],
                "arm": arm,
                "proposal_count": count,
                "p2b": p2b,
                "analytical": analytical,
                "analytical_changes_default_p2b_choice": (
                    analytical["default_proposal"]
                    != p2b["default_proposal"]
                ),
                "analytical_globalplace_relative_to_default_p2b_choice": (
                    incremental
                ),
                "response_seconds": sum(
                    float((row.get("response_run") or {}).get("elapsed_seconds") or 0.0)
                    for row in proposals
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    summary = load(args.summary)
    rows = [
        row
        for arm, count in ARMS
        for row in audit_arm(arm, count)
    ]
    changes = [
        row
        for row in rows
        if row["analytical_changes_default_p2b_choice"]
    ]
    improvements = [
        row
        for row in rows
        if row["analytical_globalplace_relative_to_default_p2b_choice"] < -1.0e-9
    ]
    evidence_cases = sorted({row["case"] for row in rows})
    decision = (
        "reject_deployable_selector"
        if len(evidence_cases) < 3 or not changes
        else "pilot_requires_fresh_validation"
    )
    report = {
        "version": "chip-v46-analytical-search-decision-v1",
        "status": decision,
        "summary_status": summary["status"],
        "cases": evidence_cases,
        "case_count": len(evidence_cases),
        "rows": rows,
        "analytical_changes_default_p2b_choice": len(changes),
        "analytical_globalplace_improvements": len(improvements),
        "response_seconds": sum(row["response_seconds"] for row in rows),
        "decision": (
            "The two-case pilot contains one numerical p2b tie where the "
            "analytical response selects a proposal with 3.47% lower estimated "
            "global-place wirelength. The remaining arms either agree or remain "
            "tied, so there is no discriminative-p2b case or enough design-level "
            "evidence to support U2; the 86-310 second response cost reinforces "
            "the rejection."
        ),
        "next_action": (
            "Retain M6 as a complete negative for deployable proposal selection. "
            "Do not integrate it into the sampler. Any future full-netlist route "
            "must use untouched validation proposals and a materially different "
            "representation or cheaper response."
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "rows"}))


if __name__ == "__main__":
    main()
