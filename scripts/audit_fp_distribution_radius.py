#!/usr/bin/env python3
"""Audit coverage and density sensitivity to the evaluation radius."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.features) as payload:
        generated = payload["generated"]
        target = payload["target"]
    distance = np.linalg.norm(
        target[:, None] - generated[None],
        axis=-1,
    )
    target_pairwise = np.linalg.norm(
        target[:, None] - target[None],
        axis=-1,
    )
    positive = target_pairwise[target_pairwise > 0]
    median_radius = float(np.median(positive))
    nearest = distance.min(axis=1)
    radii = []
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        radius = median_radius * multiplier
        radii.append(
            {
                "multiplier": multiplier,
                "radius": radius,
                "coverage": float((nearest <= radius).mean()),
                "density": float((distance <= radius).sum(axis=1).mean()),
            }
        )
    report = {
        "status": "complete",
        "conditions": int(target.shape[0]),
        "generated_samples": int(generated.shape[0]),
        "feature_dimension": int(target.shape[1]),
        "median_target_pair_radius": median_radius,
        "nearest_generated_distance": {
            "min": float(nearest.min()),
            "median": float(np.median(nearest)),
            "p90": float(np.quantile(nearest, 0.90)),
            "max": float(nearest.max()),
        },
        "radius_sweep": radii,
        "coverage_half_radius": float(np.quantile(nearest, 0.50)),
        "coverage_p90_radius": float(np.quantile(nearest, 0.90)),
        "interpretation": (
            "This smoke audit diagnoses whether zero coverage is caused by the "
            "preselected median radius. It does not replace the frozen 501-case "
            "distribution gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
