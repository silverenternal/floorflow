#!/usr/bin/env python3
"""Compare one official-100 candidate run against a fixed Trinity seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trinity", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--portfolio", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-key", default="nfe4_refine1600")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_rows(report: dict, case_key: str) -> list[dict]:
    if "results" in report:
        cases_path = report["_cases_path"]
        return load(ROOT / cases_path)[case_key]
    if "rows" not in report:
        raise ValueError("candidate report contains neither results nor portfolio rows")
    rows = []
    for row in report["rows"]:
        legalized = row["selected"]["stages"]["legalized"]
        rows.append(
            {
                "index": int(row["index"]),
                "legalized_hard_cost": float(legalized["cost"]),
                "legalized_v_rel": float(legalized["v_rel"]),
                "legalized_area_gap": float(legalized["area_gap"]),
                "legalized_b2b_hpwl_gap": float(legalized["b2b_hpwl_gap"]),
                "legalized_p2b_hpwl_gap": float(legalized["p2b_hpwl_gap"]),
                "legalized_grouping": float(legalized["grouping"]),
                "legalized_boundary": float(legalized["boundary"]),
                "legalized_rho_mae": float(legalized["rho_mae"]),
                "legalized_feasible_fraction": float(legalized["feasible"]),
            }
        )
    return rows


def summarize_candidate(report: dict, case_key: str, rows: list[dict]) -> dict:
    if "results" in report:
        return report["results"][case_key]
    count = len(rows)
    return {
        key: sum(float(row[key]) for row in rows) / count
        for key in (
            "legalized_hard_cost",
            "legalized_v_rel",
            "legalized_area_gap",
            "legalized_b2b_hpwl_gap",
            "legalized_p2b_hpwl_gap",
            "legalized_grouping",
            "legalized_boundary",
            "legalized_rho_mae",
            "legalized_feasible_fraction",
        )
    }


def main() -> None:
    args = parse_args()
    candidate_report = load(ROOT / args.candidate)
    if "results" in candidate_report:
        candidate_report["_cases_path"] = str(
            (ROOT / args.candidate).with_name(
                f"{args.candidate.stem}_cases.json"
            )
        )
    trinity = load(ROOT / args.trinity)
    candidate_rows = load_candidate_rows(candidate_report, args.case_key)
    trinity_rows = trinity["rows"]
    if len(candidate_rows) != 100 or len(trinity_rows) != 100:
        raise ValueError("official comparison requires exactly 100 cases")
    def compare(
        current: list[dict],
        baseline: list[dict],
        seed_offset: int,
    ) -> dict:
        return {
            name: paired_delta(
                current,
                baseline,
                name,
                iterations=10000,
                seed=args.seed * 100 + seed_offset + offset,
                lower_is_better=True,
            )
            for offset, name in enumerate(
                ("legalized_hard_cost", "legalized_v_rel")
            )
        }

    metrics = compare(candidate_rows, trinity_rows, 0)
    control_comparison = None
    if args.control is not None:
        control_cases = load(
            (ROOT / args.control).with_name(
                f"{args.control.stem}_cases.json"
            )
        )
        control_rows = control_cases[args.case_key]
        if len(control_rows) != 100:
            raise ValueError("official control comparison requires exactly 100 cases")
        control_comparison = compare(candidate_rows, control_rows, 100)
    portfolio_comparison = None
    if args.portfolio is not None:
        portfolio_report = load(ROOT / args.portfolio)
        portfolio_rows = load_candidate_rows(portfolio_report, args.case_key)
        if len(portfolio_rows) != 100:
            raise ValueError("official portfolio comparison requires exactly 100 cases")
        portfolio_comparison = compare(portfolio_rows, trinity_rows, 200)
    control_beaten = (
        control_comparison is not None
        and all(row["ci_high"] < 0.0 for row in control_comparison.values())
    )
    trinity_beaten = all(row["ci_high"] < 0.0 for row in metrics.values())
    portfolio_beaten = (
        portfolio_comparison is not None
        and all(row["ci_high"] < 0.0 for row in portfolio_comparison.values())
    )
    report = {
        "version": "chip-official-candidate-comparison-v2",
        "status": (
            "candidate_beats_control_and_trinity"
            if control_beaten and trinity_beaten
            else "portfolio_beats_trinity_parent_rejected"
            if portfolio_beaten and not trinity_beaten
            else "candidate_beats_control"
            if control_beaten
            else "candidate_beats_trinity"
            if trinity_beaten
            else "candidate_rejected"
        ),
        "seed": args.seed,
        "candidate": str(args.candidate),
        "trinity": str(args.trinity),
        "candidate_summary": summarize_candidate(
            candidate_report,
            args.case_key,
            candidate_rows,
        ),
        "control": str(args.control) if args.control is not None else None,
        "portfolio": str(args.portfolio) if args.portfolio is not None else None,
        "paired_vs_trinity": metrics,
        "paired_vs_control": control_comparison,
        "portfolio_vs_trinity": portfolio_comparison,
        "checks": {
            "beats_control_on_both_primaries": control_beaten,
            "beats_trinity_on_both_primaries": trinity_beaten,
            "portfolio_beats_trinity_on_both_primaries": portfolio_beaten,
        },
        "claim_boundary": (
            "One-seed official-100 matched comparison. Seed replication and "
            "external routed PPA remain mandatory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
