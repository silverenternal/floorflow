#!/usr/bin/env python3
"""Summarize controlled v45 chip training recipe screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/chip_hierarchy/v45_training"
DEFAULT_ARMS = (
    "v45_baseline_seed19_5k",
    "v45_dihedral_seed19_5k",
    "v45_hflip_seed19_5k",
    "v45_vflip_seed19_5k",
    "v45_rot180_seed19_5k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_v45_training_screen.json",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    baseline = None
    for arm in args.arms:
        path = args.root / f"{arm}.json"
        report = load(path)
        metrics = report["final_dev"]
        config = report["config"]
        row = {
            "arm": arm,
            "status": report["status"],
            "steps": config["steps"],
            "seed": config["seed"],
            "geometry_augment": config["geometry_augment"],
            "ema_decay": config["ema_decay"],
            "hierarchy_bbox_decay_steps": config["hierarchy_bbox_decay_steps"],
            "metrics": metrics,
        }
        rows.append(row)
        if baseline is None:
            baseline = metrics
    for row in rows[1:]:
        row["relative_to_baseline"] = {
            name: (
                (row["metrics"][name] - baseline[name]) / baseline[name]
                if baseline[name]
                else 0.0
            )
            for name in ("latent_mse", "rho_mae", "overlap", "group", "boundary", "area", "total")
        }
    ranked = sorted(rows, key=lambda row: row["metrics"]["total"])
    best = ranked[0]
    report = {
        "version": "chip-v45-training-screen-v1",
        "status": "complete",
        "scope": "seed 19, 5000 steps, 500-case fixed dev curve",
        "selection_metric": "lower dev total; latency and structural terms are diagnostics",
        "rows": rows,
        "ranked_arms": [row["arm"] for row in ranked],
        "best_arm": best["arm"],
        "baseline_is_best": best["arm"] == args.arms[0],
        "claim_boundary": (
            "A 5k training-curve win is a screen only. Final promotion requires "
            "fixed-refiner cost and V_rel evaluation at matched budget."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best": best["arm"],
                "baseline_is_best": report["baseline_is_best"],
                "ranked": report["ranked_arms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
