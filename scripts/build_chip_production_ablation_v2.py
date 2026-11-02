#!/usr/bin/env python3
"""Freeze the staged 5k matched global versus slot-transport screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/chip_production_v2/screen_5k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_production_ablation_v2.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def cases(path: Path) -> dict[int, dict]:
    rows = read_json(path)
    return {int(row["index"]): row for row in rows}


def paired(
    control_path: Path,
    candidate_path: Path,
    metric: str,
    *,
    iterations: int,
) -> dict:
    control = cases(control_path)
    candidate = cases(candidate_path)
    indices = sorted(set(control) & set(candidate))
    delta = np.asarray(
        [
            float(candidate[index][metric]) - float(control[index][metric])
            for index in indices
        ]
    )
    generator = np.random.default_rng(20261026)
    draws = np.asarray(
        [
            generator.choice(delta, len(delta), replace=True).mean()
            for _ in range(iterations)
        ]
    )
    control_mean = float(
        np.mean([float(control[index][metric]) for index in indices])
    )
    candidate_mean = float(
        np.mean([float(candidate[index][metric]) for index in indices])
    )
    return {
        "cases": len(indices),
        "control_mean": control_mean,
        "candidate_mean": candidate_mean,
        "paired_delta": float(delta.mean()),
        "relative_delta": (candidate_mean - control_mean) / control_mean,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def arm(root: Path, name: str) -> dict:
    report_path = root / f"{name}.json"
    official_path = root / name / "official_test100_nfe8_legalized.json"
    report = read_json(report_path)
    official = read_json(official_path)
    history = report.get("history", [])
    return {
        "training_report": str(report_path),
        "official_report": str(official_path),
        "config": report.get("config"),
        "model_parameters": report.get("model_parameters"),
        "training_seconds": (
            history[-1].get("elapsed_seconds") if history else None
        ),
        "dev": report.get("final_dev"),
        "official": {
            key: official.get(key)
            for key in (
                "cases",
                "raw_hard_cost",
                "raw_feasible_fraction",
                "legalized_hard_cost",
                "legalized_v_rel",
                "legalized_hpwl_gap",
                "legalized_area_gap",
                "legalized_feasible_fraction",
                "latent_mse",
                "rho_mae",
                "aspect_diversity_ratio",
                "active_slot_count_mean",
                "active_slot_count_min",
                "active_slot_count_max",
                "top_k_mean",
                "top_k_min",
                "top_k_max",
                "router_load_max",
                "routing_entropy_mean",
                "evaluation_seconds",
            )
        },
    }


def main() -> None:
    args = parse_args()
    arms = {
        "global_matched_seed19": arm(
            args.root,
            "global_matched_seed19",
        ),
        "slot_transport_seed19": arm(
            args.root,
            "slot_transport_seed19",
        ),
    }
    control_cases = (
        args.root
        / "global_matched_seed19"
        / "official_test100_nfe8_legalized_cases.json"
    )
    candidate_cases = (
        args.root
        / "slot_transport_seed19"
        / "official_test100_nfe8_legalized_cases.json"
    )
    comparisons = {
        metric: paired(
            control_cases,
            candidate_cases,
            metric,
            iterations=args.bootstrap,
        )
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    sparse = arms["slot_transport_seed19"]
    control = arms["global_matched_seed19"]
    parameter_relative_error = abs(
        sparse["model_parameters"] - control["model_parameters"]
    ) / control["model_parameters"]
    checks = {
        "parameter_match_within_2_percent": parameter_relative_error <= 0.02,
        "dev_non_inferior": (
            sparse["dev"]["total"] <= control["dev"]["total"] * 1.02
        ),
        "official_hard_cost_non_inferior": (
            comparisons["legalized_hard_cost"]["relative_delta"] <= 0.02
        ),
        "official_v_rel_non_inferior": (
            comparisons["legalized_v_rel"]["relative_delta"] <= 0.02
        ),
        "dynamic_active_slots": (
            sparse["official"]["active_slot_count_min"]
            < sparse["official"]["active_slot_count_max"]
        ),
        "dynamic_top_k": (
            sparse["official"]["top_k_min"]
            < sparse["official"]["top_k_max"]
        ),
    }
    accepted = all(checks.values())
    report = {
        "status": "pass" if accepted else "reject_candidate",
        "scope": "seed-19 5k matched screen; no 20k extension was run",
        "protocol": {
            "steps": 5000,
            "batch_size": 128,
            "sampling_steps": 8,
            "scorer": "score_full_fast",
            "legalizer": "scale_pack",
            "best_of_n_used": False,
            "target_derived_control": False,
        },
        "parameter_match": {
            "control": control["model_parameters"],
            "candidate": sparse["model_parameters"],
            "relative_error": parameter_relative_error,
        },
        "checks": checks,
        "arms": arms,
        "paired": comparisons,
        "decision": (
            "extend_to_20k"
            if accepted
            else "reject slot transport; keep the three-seed global-slot fallback"
        ),
        "stopped_before_20k": not accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
