#!/usr/bin/env python3
"""Aggregate matched chip ablations and sparse-routing diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_production_ablation_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def official_summary(path: Path) -> dict | None:
    report = optional(path)
    if report is None:
        return None
    return {
        key: report.get(key)
        for key in (
            "cases",
            "raw_hard_cost",
            "raw_v_rel",
            "raw_feasible_fraction",
            "legalized_hard_cost",
            "legalized_v_rel",
            "legalized_feasible_fraction",
            "latent_mse",
            "rho_mae",
            "aspect_diversity_ratio",
            "active_slot_count_mean",
            "active_slot_count_min",
            "active_slot_count_max",
            "router_budget_mean",
            "router_budget_min",
            "router_budget_max",
            "top_k_mean",
            "top_k_min",
            "top_k_max",
            "router_load_max",
            "router_load_mean",
            "routing_entropy_mean",
            "evaluation_seconds",
        )
    }


def arm(report_path: Path, official_path: Path) -> dict | None:
    report = optional(report_path)
    if report is None:
        return None
    return {
        "status": report.get("status"),
        "model_parameters": report.get("model_parameters"),
        "config": report.get("config"),
        "dev": report.get("final_dev"),
        "official": official_summary(official_path),
    }


def cases(path: Path) -> dict[int, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in report}


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
    values = np.asarray(
        [
            float(candidate[index][metric]) - float(control[index][metric])
            for index in indices
        ]
    )
    generator = np.random.default_rng(20261026)
    draws = np.asarray(
        [
            generator.choice(values, len(values), replace=True).mean()
            for _ in range(iterations)
        ]
    )
    return {
        "cases": len(indices),
        "mean_delta": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    sparse_root = ROOT / "outputs/chip_sparse/v1"
    control_root = ROOT / "outputs/chip_sparse/control_global240"
    arms = {
        "flat_token": arm(
            ROOT / "outputs/chip_large/v2_20k/dense_no_ot_huber.json",
            ROOT
            / "outputs/chip_large/v2_20k/dense_no_ot_huber/"
            "official_test100_nfe8_legalized.json",
        ),
        "global_slot_192": arm(
            ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10_20k.json",
            ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10_20k/"
            "official_test100_nfe8_legalized.json",
        ),
        "global_slot_240_matched": arm(
            control_root / "global_prod_seed19_20k.json",
            control_root
            / "global_prod_seed19_20k/"
            "official_test100_nfe8_legalized.json",
        ),
        "production_sparse": arm(
            sparse_root / "sparse_prod_seed19_20k.json",
            sparse_root
            / "sparse_prod_seed19_20k/"
            "official_test100_nfe8_legalized.json",
        ),
    }
    sparse_cases = (
        sparse_root
        / "sparse_prod_seed19_20k/"
        "official_test100_nfe8_legalized_cases.json"
    )
    control_cases = (
        control_root
        / "global_prod_seed19_20k/"
        "official_test100_nfe8_legalized_cases.json"
    )
    paired_metrics = None
    if sparse_cases.is_file() and control_cases.is_file():
        paired_metrics = {
            metric: paired(
                control_cases,
                sparse_cases,
                metric,
                iterations=args.bootstrap,
            )
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
    required = ("global_slot_240_matched", "production_sparse")
    complete = all(arms[name] is not None for name in required)
    sparse_official = (
        arms["production_sparse"]["official"]
        if arms["production_sparse"] is not None
        else None
    )
    routing_dynamic = bool(
        sparse_official
        and sparse_official["active_slot_count_min"]
        != sparse_official["active_slot_count_max"]
    )
    report = {
        "status": "complete" if complete and paired_metrics else "incomplete",
        "scope": "matched seed-19 screen; not a three-seed promotion",
        "protocol": {
            "steps": 20000,
            "batch_size": 128,
            "sampling_steps": 8,
            "scorer": "Trinity score_full_fast",
            "legalizer": "scale_pack_legalize",
            "best_of_n_used": False,
            "target_derived_control": False,
        },
        "arms": arms,
        "production_checks": {
            "parameter_count": {
                "sparse": (
                    arms["production_sparse"]["model_parameters"]
                    if arms["production_sparse"]
                    else None
                ),
                "matched_control": (
                    arms["global_slot_240_matched"]["model_parameters"]
                    if arms["global_slot_240_matched"]
                    else None
                ),
            },
            "dynamic_active_slots": routing_dynamic,
            "sparse_active_slot_range": (
                [
                    sparse_official["active_slot_count_min"],
                    sparse_official["active_slot_count_max"],
                ]
                if sparse_official
                else None
            ),
        },
        "sparse_vs_matched_control": paired_metrics,
        "promotion_rule": (
            "Promote to seeds 20/21 only if sparse improves both legalized "
            "hard cost and V_rel with intervals excluding zero."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
