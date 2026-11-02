#!/usr/bin/env python3
"""Measure pixel disagreement across saved deterministic style outputs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", type=int, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2/taxonomy.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    style_dirs = {
        style: args.label_root / f"style{style}"
        for style in args.styles
    }
    taxonomy = FloorplanTaxonomy.read(args.taxonomy)
    room_lookup = np.zeros(taxonomy.num_classes, dtype=bool)
    room_lookup[list(taxonomy.room_ids)] = True
    case_paths = sorted(style_dirs[args.styles[0]].glob("case_*.npy"))
    rows = []
    for path in case_paths:
        labels = {
            style: np.load(directory / path.name)
            for style, directory in style_dirs.items()
        }
        disagreements = []
        room_disagreements = []
        for first, second in combinations(args.styles, 2):
            difference = labels[first] != labels[second]
            disagreements.append(float(difference.mean()))
            room_mask = room_lookup[labels[first]] | room_lookup[labels[second]]
            room_disagreements.append(
                float(difference[room_mask].mean())
                if room_mask.any()
                else 0.0
            )
        rows.append(
            {
                "case": path.stem,
                "pixel_disagreement_mean": float(np.mean(disagreements)),
                "room_conditioned_disagreement_mean": float(
                    np.mean(room_disagreements)
                ),
            }
        )
    report = {
        "cases": len(rows),
        "styles": args.styles,
        "best_of_n_used": False,
        "style_selection": "user supplied; no target-derived style",
        "summary": {
            "pixel_label_disagreement_mean": float(
                np.mean(
                    [row["pixel_disagreement_mean"] for row in rows]
                )
            ),
            "pixel_label_disagreement_min": float(
                np.min(
                    [row["pixel_disagreement_mean"] for row in rows]
                )
            ),
            "pixel_label_disagreement_max": float(
                np.max(
                    [row["pixel_disagreement_mean"] for row in rows]
                )
            ),
            "room_conditioned_disagreement_mean": float(
                np.mean(
                    [
                        row["room_conditioned_disagreement_mean"]
                        for row in rows
                    ]
                )
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
