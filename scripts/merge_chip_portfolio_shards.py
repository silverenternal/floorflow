#!/usr/bin/env python3
"""Merge disjoint exact shards from the chip legalizer portfolio evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_chip_m48_portfolio import paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    shards = [load(path) for path in args.shard]
    rows = sorted(
        [row for shard in shards for row in shard["rows"]],
        key=lambda row: int(row["index"]),
    )
    if len(rows) != 100:
        raise ValueError(f"portfolio shards must cover 100 cases, got {len(rows)}")
    summary = {
        "cases": len(rows),
        "selection_counts": {
            name: sum(row["selection"] == name for row in rows)
            for name in ("baseline", "expanded", "projector32")
        },
        "solver_escalations": [
            row["index"]
            for row in rows
            if row["selected"]["legalizer"]["rung"]
            > row["baseline"]["legalizer"]["rung"]
            or row["selected"]["legalizer"]["solves"]
            > row["baseline"]["legalizer"]["solves"]
        ],
        "selected_feasible_fraction": sum(
            bool(row["selected"]["stages"]["legalized"]["feasible"])
            for row in rows
        )
        / len(rows),
    }
    summary["paired_cost"] = paired(rows, rows, "cost")
    summary["paired_v_rel"] = paired(rows, rows, "v_rel")
    summary["wall_clock_seconds"] = max(
        float(shard["summary"]["wall_clock_seconds"]) for shard in shards
    )
    report = dict(shards[0])
    report["status"] = (
        "pass"
        if not summary["solver_escalations"]
        and summary["selected_feasible_fraction"] == 1.0
        and summary["paired_cost"]["relative_delta"] < 0.0
        and summary["paired_v_rel"]["relative_delta"] < 0.0
        else "rejected"
    )
    report["summary"] = summary
    report["rows"] = rows
    report["shards"] = [str(path) for path in args.shard]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
