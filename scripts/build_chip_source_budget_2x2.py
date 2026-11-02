#!/usr/bin/env python3
"""Summarize the source-fraction by training-budget chip 2x2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

KEY = "nfe4_refine400"
METRICS = (
    "legalized_hard_cost",
    "legalized_v_rel",
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)

CELLS = {
    "2k_p050": {
        "steps": 2000,
        "shelf_probability": 0.50,
        "candidate": "outputs/reports/chip_m64_authorized_quick_refiner400_dev32_v1.json",
        "control": "outputs/reports/chip_v2_shared_control_refiner400_dev32_v1.json",
        "train": "outputs/chip_hierarchy/v2_authorized_quick/m64/candidate/m64_candidate_quick.json",
        "control_train": "outputs/chip_hierarchy/v2_authorized_quick/m50/control/m50_control_quick.json",
    },
    "2k_p075": {
        "steps": 2000,
        "shelf_probability": 0.75,
        "candidate": "outputs/reports/chip_m66_source075_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_v2_shared_control_refiner400_dev32_v1.json",
        "train": "outputs/chip_hierarchy/m66_source_075_2k_v1/candidate/m66_candidate_source075_2k.json",
        "control_train": "outputs/chip_hierarchy/v2_authorized_quick/m50/control/m50_control_quick.json",
    },
    "5k_p050": {
        "steps": 5000,
        "shelf_probability": 0.50,
        "candidate": "outputs/reports/chip_m67_source050_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m64_confirmation_control_ref400_dev32_v1.json",
        "train": "outputs/chip_hierarchy/m67_source_050_5k_v1/candidate/m67_candidate_source050_5k.json",
        "control_train": "outputs/chip_hierarchy/m64_confirmation_5k_v1/control/m64_control_confirmation5k.json",
    },
    "5k_p075": {
        "steps": 5000,
        "shelf_probability": 0.75,
        "candidate": "outputs/reports/chip_m64_confirmation_candidate_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m64_confirmation_control_ref400_dev32_v1.json",
        "train": "outputs/chip_hierarchy/m64_confirmation_5k_v1/candidate/m64_candidate_confirmation5k.json",
        "control_train": "outputs/chip_hierarchy/m64_confirmation_5k_v1/control/m64_control_confirmation5k.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_source_budget_2x2_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load(path: str | Path) -> dict:
    path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def cases(path: str | Path) -> list[dict]:
    path = ROOT / path
    path = path.with_name(f"{path.stem}_cases.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[KEY]


def relative(candidate: dict, control: dict) -> dict:
    return {
        metric: {
            "control": float(control[metric]),
            "candidate": float(candidate[metric]),
            "relative": float(candidate[metric]) / float(control[metric]) - 1.0,
        }
        for metric in METRICS
    }


def build_cell(config: dict, bootstrap: int) -> dict:
    candidate_report = load(config["candidate"])
    control_report = load(config["control"])
    candidate = candidate_report["results"][KEY]
    control = control_report["results"][KEY]
    train = load(config["train"])
    control_train = load(config["control_train"])
    metadata = relative(candidate, control)
    paired = {
        metric: paired_delta(
            cases(config["candidate"]),
            cases(config["control"]),
            metric,
            iterations=bootstrap,
            seed=19,
            lower_is_better=True,
        )
        for metric in METRICS
    }
    primary_pass = (
        metadata["legalized_hard_cost"]["relative"] <= 0.0
        and metadata["legalized_v_rel"]["relative"] <= 0.0
    )
    return {
        "steps": config["steps"],
        "shelf_probability": config["shelf_probability"],
        "status": "pass" if primary_pass else "fail",
        "feasible_fraction": candidate["legalized_feasible_fraction"],
        "relative": metadata,
        "paired": paired,
        "raw_total": train["final_dev"]["total"],
        "control_raw_total": control_train["final_dev"]["total"],
        "raw_total_relative": train["final_dev"]["total"]
        / control_train["final_dev"]["total"]
        - 1.0,
        "candidate_report": config["candidate"],
        "control_report": config["control"],
    }


def main() -> None:
    args = parse_args()
    cells = {
        name: build_cell(config, args.bootstrap)
        for name, config in CELLS.items()
    }
    report = {
        "version": "chip-source-budget-2x2-v1",
        "protocol": {
            "training_subset": "configs/chip_train_screen_5k_v1.json",
            "validation_cases": 32,
            "nfe": 4,
            "refiner_steps": 400,
            "seed": 19,
            "metric": "legalized",
        },
        "cells": cells,
        "pattern": {
            "p50_2k_pass": cells["2k_p050"]["status"] == "pass",
            "p75_2k_pass": cells["2k_p075"]["status"] == "pass",
            "p50_5k_pass": cells["5k_p050"]["status"] == "pass",
            "p75_5k_pass": cells["5k_p075"]["status"] == "pass",
        },
        "claim_boundary": (
            "A 2x2 pass identifies a robust source policy for one seed and one "
            "frozen subset. It does not establish SOTA or routed-PPA transfer."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["pattern"], indent=2))


if __name__ == "__main__":
    main()
