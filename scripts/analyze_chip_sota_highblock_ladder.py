#!/usr/bin/env python3
"""Analyze pooled and high-block slices for completed training-ladder arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Trinity/src"))

from trinity.floorplan.scoring.total import total_mean

STRATA = {
    "all": (0, 10_000),
    "low_21_60": (21, 60),
    "mid_61_90": (61, 90),
    "high_91_120": (91, 120),
    "extreme_100_120": (100, 120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-key", default="nfe4_refine400")
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--evaluation-template", default="eval_{steps}.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_rows(
    output_root: Path,
    steps: int,
    key: str,
    evaluation_template: str,
) -> list[dict]:
    evaluation = output_root / evaluation_template.format(steps=steps)
    cases_path = evaluation.with_name(f"{evaluation.stem}_cases.json")
    if not evaluation.is_file() or not cases_path.is_file():
        raise FileNotFoundError(f"missing evaluation artifacts for step {steps}")
    rows = load(cases_path)
    if key not in rows:
        raise ValueError(f"{cases_path} does not contain {key}")
    return rows[key]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"cases": 0}
    costs = [float(row["legalized_hard_cost"]) for row in rows]
    blocks = [int(row["blocks"]) for row in rows]
    return {
        "cases": len(rows),
        "block_min": min(blocks),
        "block_max": max(blocks),
        "legalized_hard_cost": total_mean(costs, blocks),
        "legalized_v_rel": float(
            np.mean([float(row["legalized_v_rel"]) for row in rows])
        ),
        "legalized_feasible_fraction": float(
            np.mean([bool(row["legalized_feasible"]) for row in rows])
        ),
    }


def stratify(rows: list[dict]) -> dict[str, dict]:
    output = {}
    for name, (lower, upper) in STRATA.items():
        selected = [
            row
            for row in rows
            if lower <= int(row["blocks"]) <= upper
        ]
        output[name] = summarize(selected)
    return output


def relative_delta(previous: dict, current: dict) -> dict:
    keys = ("legalized_hard_cost", "legalized_v_rel")
    if not all(key in previous and key in current for key in keys):
        return {}
    return {
        key: current[key] / previous[key] - 1.0
        for key in keys
    }


def main() -> None:
    args = parse_args()
    ladder = load(args.ladder_report)
    completed = (
        [int(step) for step in args.steps]
        if args.steps is not None
        else [int(row["steps"]) for row in ladder.get("steps", [])]
    )
    if not completed:
        raise ValueError("training ladder has no completed arms")

    arms = []
    previous = None
    for steps in completed:
        rows = case_rows(
            args.output_root,
            steps,
            args.result_key,
            args.evaluation_template,
        )
        slices = stratify(rows)
        deltas = {}
        if previous is not None:
            deltas = {
                name: relative_delta(previous[name], summary)
                for name, summary in slices.items()
            }
        arms.append(
            {
                "steps": steps,
                "slices": slices,
                "relative_delta_vs_previous": deltas,
                "per_case": sorted(
                    [
                        {
                            "index": int(row["index"]),
                            "blocks": int(row["blocks"]),
                            "legalized_hard_cost": float(row["legalized_hard_cost"]),
                            "legalized_v_rel": float(row["legalized_v_rel"]),
                            "legalized_feasible": bool(row["legalized_feasible"]),
                        }
                        for row in rows
                    ],
                    key=lambda row: (row["blocks"], row["index"]),
                ),
            }
        )
        previous = slices

    report = {
        "version": "chip-sota-highblock-ladder-analysis-v1",
        "status": "diagnostic_only",
        "ladder_report": str(args.ladder_report),
        "output_root": str(args.output_root),
        "result_key": args.result_key,
        "strata": {
            name: {"block_min": bounds[0], "block_max": bounds[1]}
            for name, bounds in STRATA.items()
        },
        "arms": arms,
        "claim_boundary": (
            "This analysis localizes improvement or regression by block count. "
            "It does not replace the complete dev gate, matched Trinity comparison "
            "or external confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "arms": arms}))


if __name__ == "__main__":
    main()
