#!/usr/bin/env python3
"""Join training-dynamics maps with fine-grained CubiCasa strata."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from floorflow.data.stratified_sampling import load_floorplan_strata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-map", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _normalize(name: str) -> str:
    return name.replace("__", "/", 1)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cases": len(rows),
        "regions": dict(Counter(str(row["region"]) for row in rows)),
        "confidence": _mean([float(row["confidence"]) for row in rows]),
        "variability": _mean([float(row["variability"]) for row in rows]),
        "correctness": _mean([float(row["correctness"]) for row in rows]),
        "room_precision": _mean(
            [float(row["room_precision"]) for row in rows]
        ),
        "room_recall": _mean([float(row["room_recall"]) for row in rows]),
        "opening_accuracy": _mean(
            [float(row["opening_accuracy"]) for row in rows]
        ),
        "furniture_accuracy": _mean(
            [float(row["furniture_accuracy"]) for row in rows]
        ),
        "forgetfulness": int(
            sum(int(row["forgetfulness"]) for row in rows)
        ),
    }


def main():
    args = parse_args()
    rows = []
    with args.data_map.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["_normalized_name"] = _normalize(str(payload["name"]))
            rows.append(payload)
    strata = {
        entry.name: entry
        for entry in load_floorplan_strata(args.strata, args.split)
    }
    missing = [
        str(row["_normalized_name"])
        for row in rows
        if row["_normalized_name"] not in strata
    ]
    if missing:
        raise KeyError(f"missing strata for {len(missing)} rows: {missing[:3]}")
    grouped = defaultdict(list)
    strata_groups = defaultdict(list)
    for row in rows:
        entry = strata[str(row["_normalized_name"])]
        tags = {
            "region": row["region"],
            "corpus": entry.corpus,
            "room_count": entry.room_count_bin,
            "object_density": entry.object_count_bin,
            "instance_density": entry.instance_count_bin,
            "complexity": entry.complexity_bin,
            "rare": int(entry.has_rare),
        }
        for group, value in tags.items():
            grouped[f"{group}:{value}"].append(row)
        strata_groups[entry.stratum].append(row)

    stratum_summaries = []
    for name, group_rows in strata_groups.items():
        summary = _summarize(group_rows)
        summary["stratum"] = name
        stratum_summaries.append(summary)
    by_ambiguous = sorted(
        stratum_summaries,
        key=lambda row: (
            -int(row["regions"].get("ambiguous", 0)),
            float(row["room_recall"]),
        ),
    )
    by_room_recall = sorted(
        stratum_summaries,
        key=lambda row: float(row["room_recall"]),
    )
    report = {
        "split": args.split,
        "samples": len(rows),
        "epoch": int(rows[0]["epoch"]) if rows else None,
        "coordinate_contract": {
            "confidence": "mean true-class probability inside footprint",
            "variability": "population std of confidence across snapshots",
            "correctness": "mean pixel accuracy inside footprint",
            "room_precision": "correct pixels among predicted room pixels",
            "room_recall": "correct pixels among GT room pixels",
            "region": "quantile rules from Dataset Cartography",
            "visitation": "unchanged; weights affect loss only",
        },
        "groups": {
            name: _summarize(group_rows)
            for name, group_rows in sorted(grouped.items())
        },
        "top_ambiguous_strata": by_ambiguous[:30],
        "lowest_room_recall_strata": by_room_recall[:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "samples": report["samples"],
                "regions": dict(Counter(str(row["region"]) for row in rows)),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
