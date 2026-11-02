#!/usr/bin/env python3
"""Build the target-free M48 legalizer portfolio report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)
from floorflow.evaluation.paired_bootstrap import paired_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--projector32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in payload["rows"]}


def candidate(name: str, row: dict) -> LegalizerCandidate:
    legalized = row["stages"]["legalized"]
    return LegalizerCandidate(
        name=name,
        v_rel=float(legalized["v_rel"]),
        displacement=float(
            row["displacement"]["refined_to_legalized"]["center_l2"]
        ),
        boundary=float(legalized["boundary"]),
        grouping=float(legalized["grouping"]),
        rung=int(row["legalizer"]["rung"]),
        solves=int(row["legalizer"]["solves"]),
    )


def paired(selected: list[dict], baseline: list[dict], metric: str) -> dict:
    current = [
        {
            "index": row["index"],
            "value": float(row["stages"]["legalized"][metric]),
        }
        for row in selected
    ]
    reference = [
        {
            "index": row["index"],
            "value": float(row["stages"]["legalized"][metric]),
        }
        for row in baseline
    ]
    return paired_delta(
        current,
        reference,
        "value",
        iterations=10000,
        lower_is_better=True,
    )


def main() -> None:
    args = parse_args()
    baseline = load_rows(args.baseline)
    expanded = load_rows(args.expanded)
    projector32 = load_rows(args.projector32)
    if not (baseline.keys() == expanded.keys() == projector32.keys()):
        raise ValueError("portfolio reports must contain identical case indices")
    selected = []
    choices = Counter()
    escalations = []
    for index in sorted(baseline):
        base_row = baseline[index]
        base_view = candidate("baseline", base_row)
        views = {
            "expanded": candidate("expanded", expanded[index]),
            "projector32": candidate("projector32", projector32[index]),
        }
        choice = select_legalizer_candidate(
            base_view,
            tuple(views.values()),
        )
        row = {
            "baseline": base_row,
            "expanded": expanded[index],
            "projector32": projector32[index],
        }[choice.name]
        choices[choice.name] += 1
        selected.append(row)
        if (
            int(row["legalizer"]["rung"]) > int(base_row["legalizer"]["rung"])
            or int(row["legalizer"]["solves"]) > int(base_row["legalizer"]["solves"])
        ):
            escalations.append(int(index))
    base_rows = [baseline[index] for index in sorted(baseline)]
    summary = {
        "cases": len(selected),
        "selections": dict(choices),
        "solver_escalations": escalations,
        "baseline_cost": sum(
            row["stages"]["legalized"]["cost"] for row in base_rows
        )
        / len(base_rows),
        "selected_cost": sum(
            row["stages"]["legalized"]["cost"] for row in selected
        )
        / len(selected),
        "baseline_v_rel": sum(
            row["stages"]["legalized"]["v_rel"] for row in base_rows
        )
        / len(base_rows),
        "selected_v_rel": sum(
            row["stages"]["legalized"]["v_rel"] for row in selected
        )
        / len(selected),
        "paired_cost": paired(selected, base_rows, "cost"),
        "paired_v_rel": paired(selected, base_rows, "v_rel"),
    }
    report = {
        "version": "chip-m48-legalizer-portfolio-v1",
        "status": (
            "quick_pass_bounded_portfolio"
            if not escalations
            and summary["paired_cost"]["relative_delta"] < 0.0
            and summary["paired_v_rel"]["relative_delta"] < 0.0
            else "rejected"
        ),
        "selection_contract": {
            "uses_target": False,
            "order": [
                "v_rel",
                "refined_to_legalized_center_l2",
                "boundary",
                "grouping",
                "candidate_name",
            ],
            "eligibility": "rung <= baseline rung and solves <= baseline solves",
        },
        "budget": {
            "baseline": {
                "nfe": 4,
                "refiner_steps": 400,
                "legalizer_calls": 1,
            },
            "portfolio": {
                "nfe": 8,
                "refiner_steps": 800,
                "legalizer_calls": 3,
            },
        },
        "inputs": {
            "baseline": str(args.baseline),
            "expanded": str(args.expanded),
            "projector32": str(args.projector32),
        },
        "summary": summary,
        "rows": selected,
        "boundary": (
            "Target-free deployable portfolio diagnostic. It consumes two "
            "trajectories and must report the doubled generation budget."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
