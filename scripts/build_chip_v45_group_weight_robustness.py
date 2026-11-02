#!/usr/bin/env python3
"""Aggregate the dev-only group-weight robustness screen across model seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)
METRICS = ("legalized_hard_cost", "legalized_v_rel", "legalized_feasible_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/chip_v45_group_weight_screen.json",
    )
    parser.add_argument("--seed19-screen", type=Path, default=ROOT / "outputs/reports/chip_v45_refiner_weight_screen.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/chip_v45_group_weight_robustness.json")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tag(weight: float) -> str:
    return str(weight).replace(".", "p")


def report_path(seed: int, weight: float) -> Path:
    if seed == 21 and tag(weight) == "2p0":
        return ROOT / "outputs/reports/chip_v45_baseline_seed21_20k_group2_dev.json"
    suffix = {
        "1p0": "1",
        "2p0": "2",
    }.get(tag(weight), tag(weight))
    return ROOT / f"outputs/reports/chip_v45_weight_seed{seed}_g{suffix}_dev.json"


def report_summary(path: Path) -> dict[str, float]:
    report = load(path)
    keys = list(report["results"])
    if len(keys) != 1:
        raise ValueError(f"expected one result key in {path}, found {keys}")
    result = report["results"][keys[0]]
    return {metric: float(result[metric]) for metric in METRICS}


def seed19_existing(path: Path, weight: float) -> dict[str, float]:
    arms = load(path)["arms"]
    arms = {key.replace(".", "p"): value for key, value in arms.items()}
    aliases = {
        "1p0": "linear",
        "1p25": "group1p25",
        "1p5": "group1p5",
        "2p0": "group2",
    }
    return {
        metric: float(arms[aliases[tag(weight)]][metric])
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    config = load(args.config)
    weights = [float(value) for value in config["group_weights"]]
    results: dict[str, dict[str, dict[str, float]]] = {
        tag(weight): {} for weight in weights
    }
    for weight in weights:
        for seed in SEEDS:
            if seed == 19 and tag(weight) != "1p25":
                summary = seed19_existing(args.seed19_screen, weight)
            else:
                summary = report_summary(report_path(seed, weight))
            results[tag(weight)][str(seed)] = summary
    relative: dict[str, dict[str, float]] = {tag(weight): {} for weight in weights}
    for weight in weights[1:]:
        for seed in SEEDS:
            base = results["1p0"][str(seed)]
            row = results[tag(weight)][str(seed)]
            relative[tag(weight)][str(seed)] = {
                "cost": row["legalized_hard_cost"] / base["legalized_hard_cost"] - 1.0,
                "v_rel": row["legalized_v_rel"] / base["legalized_v_rel"] - 1.0,
            }
    arms = {}
    for weight in weights[1:]:
        values = relative[tag(weight)]
        costs = [value["cost"] for value in values.values()]
        v_rels = [value["v_rel"] for value in values.values()]
        feasible = all(
            results[tag(weight)][str(seed)]["legalized_feasible_fraction"] == 1.0
            for seed in SEEDS
        )
        eligible = bool(
            np.mean(costs) < 0.0
            and np.mean(v_rels) < 0.0
            and feasible
        )
        arms[tag(weight)] = {
            "per_seed_summary": results[tag(weight)],
            "relative_to_group1": values,
            "mean_relative_cost": float(np.mean(costs)),
            "mean_relative_v_rel": float(np.mean(v_rels)),
            "max_relative_cost_regression": float(np.max(costs)),
            "max_relative_v_rel_regression": float(np.max(v_rels)),
            "feasible_all_seeds": feasible,
            "admissible": eligible,
        }
    eligible = [name for name, row in arms.items() if row["admissible"]]
    best = min(
        eligible,
        key=lambda name: (arms[name]["mean_relative_cost"], arms[name]["mean_relative_v_rel"]),
        default=None,
    )
    report = {
        "version": "chip-v45-group-weight-robustness-v1",
        "status": "pass" if best is not None else "reject",
        "scope": "Frozen 32-case dev; one shared weight across candidate training seeds",
        "baseline": "group=1.0 boundary=2.0",
        "results": results,
        "arms": arms,
        "eligible_weights": eligible,
        "selected_weight": best,
        "checks": {
            "all_seed_weight_pairs_present": all(
                len(results[tag(weight)]) == len(SEEDS) for weight in weights
            ),
            "all_feasible": all(
                row["feasible_all_seeds"] for row in arms.values()
            ),
            "nonbaseline_weight_admissible": best is not None,
        },
        "claim_boundary": (
            "This is a dev-only recipe selection. It cannot establish matched-compute "
            "SOTA until a same-protocol Official-100 confirmation is run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_weight": best, "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
