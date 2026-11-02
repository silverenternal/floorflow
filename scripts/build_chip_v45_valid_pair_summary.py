#!/usr/bin/env python3
"""Summarize only v45 pairs that passed legalized pair-integrity checks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppa",
        type=Path,
        default=ROOT / "outputs/reports/chip_external_ppa_v45_partial.json",
    )
    parser.add_argument(
        "--pair-integrity",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_external_pair_integrity.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_external_valid_pairs_summary.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def leave_one_out_mean(values: list[float]) -> dict[str, float]:
    means = [
        float((sum(values) - value) / (len(values) - 1))
        for value in values
    ]
    return {"min": min(means), "max": max(means)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wire = [
        float(row["detailedroute__route__wirelength"]["relative"])
        for row in rows
    ]
    congestion = [
        float(row["globalplace__gpl__routability__congestion"]["relative"])
        for row in rows
    ]
    setup_delta = [
        float(row["finish__timing__setup__ws"]["candidate"])
        - float(row["finish__timing__setup__ws"]["control"])
        for row in rows
    ]
    power = [
        float(row["finish__power__total"]["relative"])
        for row in rows
    ]
    area = [
        float(row["finish__design__instance__area__stdcell"]["relative"])
        for row in rows
    ]
    return {
        "case_count": len(rows),
        "cases": [row["case"] for row in rows],
        "routed_wirelength_relative": distribution(wire),
        "routed_wirelength_leave_one_out_mean": leave_one_out_mean(wire),
        "wirelength_wins_ties_losses": {
            "wins": sum(value < -1.0e-9 for value in wire),
            "ties": sum(abs(value) <= 1.0e-9 for value in wire),
            "losses": sum(value > 1.0e-9 for value in wire),
        },
        "congestion_relative": distribution(congestion),
        "finish_setup_ws_delta_ns": distribution(setup_delta),
        "power_relative": distribution(power),
        "stdcell_area_relative": distribution(area),
        "drc_zero": all(
            row.get("candidate_drc") == 0 and row.get("control_drc") == 0
            for row in rows
        ),
    }


def per_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case": row["case"],
            "source_family": row.get("source_family"),
            "seed": int(row["seed"]),
            "routed_wirelength": row["detailedroute__route__wirelength"],
            "congestion": row["globalplace__gpl__routability__congestion"],
            "finish_setup_ws": row["finish__timing__setup__ws"],
            "power": row["finish__power__total"],
            "stdcell_area": row["finish__design__instance__area__stdcell"],
            "candidate_drc": row.get("candidate_drc"),
            "control_drc": row.get("control_drc"),
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    ppa = load(args.ppa)
    integrity = load(args.pair_integrity)
    completed = {
        row["case"]: row
        for row in ppa["comparisons"]
        if row.get("status") == "completed"
    }
    valid_cases = list(integrity["valid_completed_cases"])
    invalid_cases = list(integrity["invalid_completed_cases"])
    missing_cases = [
        row["case"]
        for row in integrity["cases"]
        if row["ppa_status"] != "completed"
    ]
    valid_rows = [completed[case] for case in valid_cases if case in completed]
    invalid_rows = [completed[case] for case in invalid_cases if case in completed]
    all_rows = [completed[case] for case in valid_cases + invalid_cases if case in completed]
    valid_summary = summarize(valid_rows)
    valid_summary["per_case"] = per_case_rows(valid_rows)
    report = {
        "version": "chip-v45-valid-pair-summary-v1",
        "status": "complete_for_valid_pairs",
        "scope": (
            "Only candidate/control pairs with distinct legalized TCL hashes. "
            "The result remains external development evidence because the v45 "
            "set is observed and incomplete."
        ),
        "inputs": {
            "ppa": str(args.ppa.relative_to(ROOT)),
            "pair_integrity": str(args.pair_integrity.relative_to(ROOT)),
        },
        "counts": {
            "valid_completed": len(valid_rows),
            "invalid_completed": len(invalid_rows),
            "missing_or_failed": len(missing_cases),
        },
        "valid": valid_summary,
        "invalid_collapsed_completed": summarize(invalid_rows),
        "all_completed_with_collapsed_included": summarize(all_rows),
        "missing_or_failed_cases": missing_cases,
        "interpretation": (
            "Collapsed pairs contribute artificial zero differences. The valid "
            "subset is the only v45 routed-PPA model comparison, and it remains "
            "negative on mean routed wirelength."
        ),
        "claim_boundary": (
            "No external SOTA or subgroup claim follows from eight valid observed "
            "designs. The original v45 endpoints and collapsed pairs remain archived."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid_cases": len(valid_rows),
                "wirelength_mean": report["valid"]["routed_wirelength_relative"]["mean"],
                "wirelength_wins_ties_losses": report["valid"]["wirelength_wins_ties_losses"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
