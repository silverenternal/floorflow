#!/usr/bin/env python3
"""Summarize matched M64 training-seed replication at 2k/ref400."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

KEY = "nfe4_refine400"
SEEDS = {
    19: {
        "candidate": "outputs/reports/chip_m69_seed19_candidate_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m69_seed19_control_ref400_dev32_v1.json",
        "candidate_train": (
            "outputs/chip_hierarchy/m69_training_seed_replication_v1/seed19/"
            "candidate/m69_seed19_candidate_2k.json"
        ),
        "control_train": (
            "outputs/chip_hierarchy/m69_training_seed_replication_v1/seed19/"
            "control/m69_seed19_control_2k.json"
        ),
    },
    20: {
        "candidate": "outputs/reports/chip_m69_seed20_candidate_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m69_seed20_control_ref400_dev32_v1.json",
        "candidate_train": (
            "outputs/chip_hierarchy/m69_training_seed_replication_v1/seed20/"
            "candidate/m69_seed20_candidate_2k.json"
        ),
        "control_train": (
            "outputs/chip_hierarchy/m69_training_seed_replication_v1/seed20/"
            "control/m69_seed20_control_2k.json"
        ),
    },
    21: {
        "candidate": "outputs/reports/chip_m64_authorized_quick_refiner400_dev32_v1.json",
        "control": "outputs/reports/chip_v2_shared_control_refiner400_dev32_v1.json",
        "candidate_train": (
            "outputs/chip_hierarchy/v2_authorized_quick/m64/candidate/"
            "m64_candidate_quick.json"
        ),
        "control_train": (
            "outputs/chip_hierarchy/v2_authorized_quick/m50/control/"
            "m50_control_quick.json"
        ),
    },
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m69_training_seed_replication_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def case_rows(path: str | Path) -> list[dict]:
    path = ROOT / path
    cases = path.with_name(f"{path.stem}_cases.json")
    return json.loads(cases.read_text(encoding="utf-8"))[KEY]


def summarize_seed(config: dict, iterations: int) -> dict:
    candidate_report = load(config["candidate"])
    control_report = load(config["control"])
    candidate = candidate_report["results"][KEY]
    control = control_report["results"][KEY]
    relative = {
        metric: float(candidate[metric]) / float(control[metric]) - 1.0
        for metric in METRICS
    }
    paired = {
        metric: paired_delta(
            case_rows(config["candidate"]),
            case_rows(config["control"]),
            metric,
            iterations=iterations,
            seed=19,
            lower_is_better=True,
        )
        for metric in METRICS
    }
    candidate_train = load(config["candidate_train"])
    control_train = load(config["control_train"])
    return {
        "candidate": config["candidate"],
        "control": config["control"],
        "relative": relative,
        "paired": paired,
        "candidate_absolute": candidate,
        "control_absolute": control,
        "candidate_raw_total": candidate_train["final_dev"]["total"],
        "control_raw_total": control_train["final_dev"]["total"],
        "raw_total_relative": (
            candidate_train["final_dev"]["total"]
            / control_train["final_dev"]["total"]
            - 1.0
        ),
        "feasibility_both_one": (
            candidate["legalized_feasible_fraction"] == 1.0
            and control["legalized_feasible_fraction"] == 1.0
        ),
        "joint_primary_improves": (
            relative["legalized_hard_cost"] < 0.0
            and relative["legalized_v_rel"] < 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    seeds = {
        str(seed): summarize_seed(config, args.bootstrap)
        for seed, config in SEEDS.items()
    }
    checks = {
        "all_feasible": all(item["feasibility_both_one"] for item in seeds.values()),
        "joint_primary_in_at_least_two_seeds": sum(
            item["joint_primary_improves"] for item in seeds.values()
        )
        >= 2,
        "cost_improves_in_at_least_two_seeds": sum(
            item["relative"]["legalized_hard_cost"] < 0.0
            for item in seeds.values()
        )
        >= 2,
        "v_rel_improves_in_at_least_two_seeds": sum(
            item["relative"]["legalized_v_rel"] < 0.0
            for item in seeds.values()
        )
        >= 2,
    }
    report = {
        "version": "chip-m69-training-seed-replication-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "protocol": {
            "steps": 2000,
            "nfe": 4,
            "refiner_steps": 400,
            "inference_seed": 19,
            "cases": 32,
        },
        "checks": checks,
        "seeds": seeds,
        "claim_boundary": (
            "A pass selects M64 for Official-100 and equal-budget Trinity "
            "certification. It remains a frozen-subset result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
