#!/usr/bin/env python3
"""Freeze and audit the pure-chip metric and legality contract."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        type=Path,
        default=ROOT / "outputs/reports/chip_token_flow_component_rows_v2.json",
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=ROOT / "outputs/reports/chip_trinity_token_three_seed_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_pure_metric_contract_v2.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rate(rows: list[dict], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "raw_hard_penalty_rate": rate(
            rows,
            lambda row: float(
                row.get(
                    "raw_hard_cost",
                    row.get("legalized_hard_cost", row.get("cost", 0.0)),
                )
            )
            >= 9.999,
        ),
        "raw_overlap_case_rate": rate(
            rows,
            lambda row: int(
                row.get(
                    "raw_overlap_pairs",
                    row.get("legalized_overlap_pairs", row.get("overlap_pairs", 0)),
                )
            )
            > 0,
        ),
        "raw_area_violation_case_rate": rate(
            rows,
            lambda row: int(
                row.get(
                    "raw_area_violations",
                    row.get("legalized_area_violations", 0),
                )
            )
            > 0,
        ),
        "legalized_hard_penalty_rate": rate(
            rows,
            lambda row: float(row["legalized_hard_cost"]) >= 9.999,
        ),
        "legalized_feasible_fraction_inferred": 1.0
        - rate(
            rows,
            lambda row: float(row["legalized_hard_cost"]) >= 9.999,
        ),
        "legalized_overlap_case_rate": rate(
            rows,
            lambda row: int(row.get("legalized_overlap_pairs", 0)) > 0,
        ),
        "legalized_area_violation_case_rate": rate(
            rows,
            lambda row: int(row.get("legalized_area_violations", 0)) > 0,
        ),
        "legalized_aspect_violation_case_rate": rate(
            rows,
            lambda row: int(row.get("legalized_aspect_violations", 0)) > 0,
        ),
        "legalized_fixed_violation_case_rate": rate(
            rows,
            lambda row: int(row.get("legalized_fixed_violations", 0)) > 0,
        )
        if any("legalized_fixed_violations" in row for row in rows)
        else None,
        "legalized_preplaced_violation_case_rate": rate(
            rows,
            lambda row: int(row.get("legalized_preplaced_violations", 0)) > 0,
        )
        if any("legalized_preplaced_violations" in row for row in rows)
        else None,
        "legalized_cost_mean": mean(rows, "legalized_hard_cost"),
        "legalized_v_rel_mean": mean(rows, "legalized_v_rel"),
        "legalized_b2b_hpwl_gap_mean": mean(rows, "legalized_b2b_hpwl_gap"),
        "legalized_p2b_hpwl_gap_mean": mean(rows, "legalized_p2b_hpwl_gap"),
        "legalized_area_gap_mean": mean(rows, "legalized_area_gap"),
    }


def v_rel_errors(rows: list[dict]) -> list[float]:
    errors = []
    for row in rows:
        slots = max(1, int(row.get("legalized_soft_slots", 0)))
        reconstructed = (
            int(row.get("legalized_grouping", 0))
            + int(row.get("legalized_mib", 0))
            + int(row.get("legalized_boundary", 0))
        ) / slots
        errors.append(abs(reconstructed - float(row["legalized_v_rel"])))
    return errors


def main() -> None:
    args = parse_args()
    component_payload = read(args.components)
    candidate_summary = read(args.candidate_summary)
    if component_payload["status"] != "complete":
        raise ValueError("component rows are incomplete")
    grouped = defaultdict(list)
    for row in component_payload["rows"]:
        grouped[(row["arm"], row["stage"])].append(row)
    trinity_rows = []
    for seed in (19, 20, 21):
        trinity_rows.extend(
            read(
                ROOT
                / f"outputs/reports/trinity_d192_component_seed{seed}_v1.json"
            )["rows"]
        )
    summaries = {
        f"{arm}:{stage}": summarize(rows)
        for (arm, stage), rows in sorted(grouped.items())
    }
    summaries["trinity:reproducible"] = summarize(trinity_rows)
    errors = v_rel_errors(component_payload["rows"] + trinity_rows)
    legalized_rows = [
        row
        for row in component_payload["rows"] + trinity_rows
        if "legalized_hard_cost" in row
    ]
    checks = {
        "all_component_sets_complete": all(
            summary["cases"] > 0 for summary in summaries.values()
        ),
        "v_rel_reconstructs_from_three_components": max(errors, default=0.0)
        < 1.0e-6,
        "all_scored_legalized_layouts_below_penalty": all(
            float(row["legalized_hard_cost"]) < 9.999
            for row in legalized_rows
        ),
        "current_candidate_feasible_fraction_is_one": float(
            candidate_summary["seeds"]["19"]["endpoints"]["refined"][
                "candidate"
            ]["legalized_feasible_fraction"]
        )
        == 1.0,
    }
    report = {
        "status": "pass" if all(checks.values()) else "contract_repair",
        "scope": "pure chip floorplanning; FloorSet official-100 semantics",
        "contract": {
            "hard_penalty": 10.0,
            "hard_constraints": [
                "no overlap",
                "soft block area within 1%",
                "fixed block shape unchanged",
                "preplaced block box unchanged",
                "aspect bounds when provided",
            ],
            "cost_feasible": (
                "(1 + 0.5 * (max(hpwl_gap, 0) + max(area_gap, 0))) "
                "* exp(2 * V_rel)"
            ),
            "v_rel": (
                "(grouping + MIB + boundary) / N_soft; HPWL and "
                "area_gap are never included"
            ),
            "v_rel_denominator": (
                "boundary-coded blocks + sum(cluster_size-1) + "
                "sum(MIB_group_size-1)"
            ),
            "primary_metrics": [
                "legalized_hard_cost",
                "legalized_v_rel",
                "legalized_b2b_hpwl_gap",
                "legalized_p2b_hpwl_gap",
                "legalized_area_gap",
            ],
            "legality_metrics": [
                "hard_cost_equal_10_rate",
                "overlap_pairs",
                "area_violations",
                "fixed_violations",
                "preplaced_violations",
                "aspect_violations",
            ],
            "unavailable_fields": [
                "opening channel/metric",
                "port direction",
                "halo geometry",
                "routing congestion map",
            ],
            "MIB_semantics": (
                "FloorSet multi-instantiation constraint: member blocks "
                "must share one (w,h)"
            ),
            "boundary_semantics": (
                "coded blocks touch required left/right/top/bottom "
                "edges of the global layout bounding box"
            ),
        },
        "summaries": summaries,
        "reconciliation": {
            "v_rel_max_abs_error": max(errors, default=0.0),
            "legalized_rows": len(legalized_rows),
        },
        "checks": checks,
        "notes": [
            "Existing v2 component rows predate explicit fixed/preplaced counts, but hard cost < 10 and feasible_fraction=1 imply both passed.",
            "New score_diagnostics rows serializes fixed and preplaced violation counts for all future runs.",
            "Raw overlap is a diagnostic; raw feasibility is not an official endpoint.",
            "FloorSet official cases have no fixed outline, opening, direction, halo or congestion field.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
