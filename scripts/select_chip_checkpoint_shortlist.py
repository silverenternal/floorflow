"""Select a frozen chip checkpoint from legalized shortlist metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorflow.chip_flow.checkpoint_selection import (
    rank_legalized_candidates,
)
from floorflow.chip_flow.training_runtime import write_json_atomic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--legalized-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    legalized = json.loads(args.legalized_report.read_text(encoding="utf-8"))
    shortlist = training.get("checkpoint_selection", {}).get(
        "shortlist_candidates",
        [],
    )
    path_by_step = {
        int(candidate["step"]): str(candidate["checkpoint"])
        for candidate in shortlist
    }
    rows = legalized.get("rows", legalized)
    candidates = []
    for row in rows:
        step = int(row["step"])
        if step in path_by_step:
            candidates.append(
                {
                    **row,
                    "checkpoint": path_by_step[step],
                    "raw_total": next(
                        float(candidate["raw_total"])
                        for candidate in shortlist
                        if int(candidate["step"]) == step
                    ),
                }
            )
    ranked = rank_legalized_candidates(candidates)
    if not ranked:
        raise ValueError("no overlapping shortlist candidates were found")
    report = {
        "version": "chip-checkpoint-selection-v1",
        "ranking": ranked,
        "selected": ranked[0],
        "policy": (
            "feasible, legalized cost, legalized V_rel, displacement, "
            "rung, raw total, step"
        ),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
