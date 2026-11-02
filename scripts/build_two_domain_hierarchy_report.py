#!/usr/bin/env python3
"""Assemble the route-B chip and house dynamic-hierarchy comparison."""

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
        default=ROOT / "outputs/reports/two_domain_dynamic_hierarchy_v1.json",
    )
    parser.add_argument(
        "--hierarchical-root",
        type=Path,
        default=ROOT / "outputs/chip_hierarchical/v2_matched",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path: Path) -> dict | list | None:
    return read(path) if path.is_file() else None


def official_metrics(path: Path) -> dict | None:
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
        )
    }


def rows_by_index(path: Path) -> dict[int, dict]:
    return {int(row["index"]): row for row in read(path)}


def paired_delta(
    control_path: Path,
    candidate_path: Path,
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict:
    control = rows_by_index(control_path)
    candidate = rows_by_index(candidate_path)
    indices = sorted(set(control) & set(candidate))
    values = np.asarray(
        [
            float(candidate[index][metric]) - float(control[index][metric])
            for index in indices
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(iterations)]
    )
    return {
        "cases": len(indices),
        "mean_delta": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    dense_report = read(ROOT / "outputs/chip_large/v2_20k/dense_no_ot_huber.json")
    global_report = read(
        ROOT / "outputs/chip_slot/global_v1/global_slot_overlap10_20k.json"
    )
    global_seed_report = read(
        ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json"
    )
    hierarchy_report_path = (
        args.hierarchical_root / "hierarchical_slot_overlap10_20k.json"
    )
    hierarchy_report = optional(hierarchy_report_path)
    trinity = read(ROOT / "outputs/reports/trinity_d192_official_test100.json")
    global_official_path = (
        ROOT
        / "outputs/chip_slot/global_v1/global_slot_overlap10_20k/"
        "official_test100_nfe8_legalized.json"
    )
    global_cases_path = global_official_path.with_name(
        "official_test100_nfe8_legalized_cases.json"
    )
    hierarchy_official_path = (
        args.hierarchical_root
        / "hierarchical_slot_overlap10_20k"
        / "official_test100_nfe8_legalized.json"
    )
    hierarchy_cases_path = hierarchy_official_path.with_name(
        "official_test100_nfe8_legalized_cases.json"
    )
    chip_rows = {
        "flat_token_flow": {
            "model_parameters": dense_report.get("model_parameters"),
            "dev": dense_report.get("final_dev"),
            "official": official_metrics(
                ROOT
                / "outputs/chip_large/v2_20k/dense_no_ot_huber/"
                "official_test100_nfe8_legalized.json"
            ),
        },
        "global_slot_flow": {
            "model_parameters": global_report.get("model_parameters"),
            "dev": global_report.get("final_dev"),
            "official": official_metrics(global_official_path),
        },
        "hierarchical_slot_flow": {
            "model_parameters": (
                hierarchy_report.get("model_parameters")
                if isinstance(hierarchy_report, dict)
                else None
            ),
            "dev": (
                hierarchy_report.get("final_dev")
                if isinstance(hierarchy_report, dict)
                else None
            ),
            "official": official_metrics(hierarchy_official_path),
        },
        "trinity_d192_external": {
            "samples": 1,
            "official": {
                "cases": trinity["samples"],
                "mean_cost": trinity["mean_cost"],
                "feasible_fraction": trinity["feasible_fraction"],
            },
            "note": (
                "External FloorSet baseline, not a matched architecture ablation; "
                "its checkpoint and scorer must be reported explicitly."
            ),
        },
    }
    paired = None
    if global_cases_path.is_file() and hierarchy_cases_path.is_file():
        paired = {
            metric: paired_delta(
                global_cases_path,
                hierarchy_cases_path,
                metric,
                iterations=args.bootstrap,
                seed=20261026 + offset,
            )
            for offset, metric in enumerate(
                ("legalized_hard_cost", "legalized_v_rel")
            )
        }
    house = read(
        ROOT / "outputs/reports/plan_c_house_global_slot_seeds_v1.json"
    )
    global_history = next(
        row["candidate"]["history"]
        for row in global_seed_report["seeds"]
        if int(row["seed"]) == 19
    )
    curve_keys = (
        "step",
        "total",
        "latent_mse",
        "overlap",
        "group",
        "boundary",
        "area",
    )

    def curve(history: list[dict]) -> list[dict]:
        return [
            {key: row.get(key) for key in curve_keys}
            for row in history
            if int(row["step"]) >= 6000
        ]

    complete = (
        chip_rows["hierarchical_slot_flow"]["official"] is not None
        and paired is not None
    )
    hierarchy_promotable = bool(
        paired
        and paired["legalized_hard_cost"]["mean_delta"] < 0.0
        and paired["legalized_v_rel"]["mean_delta"] < 0.0
    )
    report = {
        "status": "complete" if complete else "incomplete",
        "claim_role": "cross-domain dynamic hierarchy ablations",
        "protocol": {
            "chip_dataset": "FloorSet-Lite official test100",
            "chip_checkpoint_selection": "dev only; test evaluated once",
            "sampling_steps": 8,
            "scorer": "Trinity score_full_fast",
            "legalizer": "Trinity scale_pack_legalize",
            "best_of_n_used": False,
            "target_derived_control": False,
            "house_dataset": "CubiCasa held-out test",
        },
        "chip": {
            "rows": chip_rows,
            "hierarchical_vs_global_paired": paired,
            "training_curves": {
                "global_slot_flow": curve(global_history),
                "hierarchical_slot_flow": (
                    curve(hierarchy_report["history"])
                    if isinstance(hierarchy_report, dict)
                    else []
                ),
            },
        },
        "route_b_promotion": {
            "status": "pass" if hierarchy_promotable else "fail",
            "rule": (
                "Hierarchy must improve both paired legalized hard cost and "
                "V_rel over global slot."
            ),
        },
        "house": {
            "summary": house["summary"],
            "gate": house["gate"],
            "seeds": house["seeds"],
        },
        "interpretation_gate": (
            "Promote dynamic hierarchy only if the official chip endpoint and "
            "paired intervals support it; otherwise keep route A as primary."
        ),
        "sources": [
            str(path.relative_to(ROOT))
            for path in (
                hierarchy_report_path,
                hierarchy_official_path,
                hierarchy_cases_path,
            )
            if path.is_file()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
