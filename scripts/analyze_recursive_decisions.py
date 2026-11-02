#!/usr/bin/env python3
"""Join learned recursion decisions with CAD metrics and data strata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.data.stratified_sampling import load_floorplan_strata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(rows):
    return {
        "cases": len(rows),
        "room_macro_iou": float(
            np.mean([row["room_macro_iou"] for row in rows])
        ),
        "instance_geometric_iou": float(
            np.mean([row["instance_geometric_iou"] for row in rows])
        ),
        "instance_type_accuracy": float(
            np.mean([row["instance_type_accuracy"] for row in rows])
        ),
        "fpeval_room_polygon_f1": float(
            np.mean([row["fpeval_room_polygon_f1"] for row in rows])
        ),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.per_case.read_text(encoding="utf-8"))
    metrics = {
        row["name"].replace("__", "/", 1): row
        for row in payload["cases"]
    }
    entries = load_floorplan_strata(args.strata, args.split)
    stop = None if args.limit is None else args.offset + args.limit
    entries = entries[args.offset:stop]
    action_rows = defaultdict(list)
    depth_rows = defaultdict(list)
    stratum_rows = defaultdict(list)
    for entry in entries:
        row = metrics[entry.name]
        decision = row.get("recursive_decision")
        if decision is None:
            raise ValueError(f"missing recursion decision for {entry.name}")
        action_rows[("direct", "recursive", "split")[decision["action"]]].append(
            row
        )
        depth = max(decision["depths"], default=0)
        depth_rows[depth].append(row)
        stratum_rows[entry.stratum].append(row)
    report = {
        "split": args.split,
        "cases": len(entries),
        "best_of_n_used": False,
        "target_used_for_model_decisions": False,
        "by_action": {
            action: summarize(rows)
            for action, rows in sorted(action_rows.items())
        },
        "by_depth": {
            str(depth): summarize(rows)
            for depth, rows in sorted(depth_rows.items())
        },
        "by_stratum": {
            stratum: summarize(rows)
            for stratum, rows in sorted(stratum_rows.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("cases", "by_action", "by_depth")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
