#!/usr/bin/env python3
"""Join fine-grained data strata with per-case model metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorflow.data.stratified_sampling import load_floorplan_strata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _normalize(name: str) -> str:
    return name.replace("__", "/")


def _summarize(rows):
    metrics = {
        "room_macro_iou": "room_macro_iou",
        "room_support_iou": "room_support_iou",
        "instance_geometric_iou": "instance_geometric_iou",
        "instance_room_f1": "instance_room_f1",
        "instance_type_accuracy": "instance_type_accuracy",
        "matched_class_constrained_iou": "matched_class_constrained_iou",
        "class_constrained_iou": "class_constrained_iou",
        "coverage": "coverage",
        "boundary_f1": "boundary_f1",
        "fpeval_room_polygon_f1": "fpeval_room_polygon_f1",
        "fpeval_wall_f1": "fpeval_wall_f1",
        "fpeval_opening_f1": "fpeval_opening_f1",
        "fpeval_watertight": "fpeval_watertight",
    }
    available = {
        output: source
        for output, source in metrics.items()
        if all(source in row for row in rows)
    }
    return {
        "cases": len(rows),
        **{
            key: float(np.mean([float(row[source]) for row in rows]))
            for key, source in available.items()
        },
    }


def main():
    args = parse_args()
    payload = json.loads(args.per_case.read_text(encoding="utf-8"))
    metrics = {
        _normalize(row["name"]): row
        for row in payload["cases"]
    }
    entries = load_floorplan_strata(args.strata, args.split)
    end = None if args.limit is None else args.offset + args.limit
    entries = entries[args.offset:end]
    grouped = defaultdict(list)
    for entry in entries:
        row = metrics.get(entry.name)
        if row is None:
            raise KeyError(f"missing metrics for {entry.name}")
        tags = {
            "corpus": entry.corpus,
            "room_count": str(entry.room_count_bin),
            "object_density": str(entry.object_count_bin),
            "complexity": str(entry.complexity_bin),
            "instance_density": str(entry.instance_count_bin),
            "rare": str(int(entry.has_rare)),
            "stratum": entry.stratum,
            "fine_stratum": entry.fine_stratum or entry.stratum,
        }
        for group, value in tags.items():
            grouped[f"{group}:{value}"].append(row)
    report = {
        "split": args.split,
        "cases": len(entries),
        "groups": {
            name: _summarize(rows)
            for name, rows in sorted(grouped.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: group
                for name, group in report["groups"].items()
                if ":" in name and name.split(":", 1)[0] == "corpus"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
