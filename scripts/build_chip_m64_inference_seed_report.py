#!/usr/bin/env python3
"""Certify the M64 2k/p50 quick gate across inference seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

KEY = "nfe4_refine400"
REPORTS = {
    19: {
        "candidate": "outputs/reports/chip_m64_authorized_quick_refiner400_dev32_v1.json",
        "control": "outputs/reports/chip_v2_shared_control_refiner400_dev32_v1.json",
    },
    20: {
        "candidate": "outputs/reports/chip_m64_seed20_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m64_control_seed20_ref400_dev32_v1.json",
    },
    21: {
        "candidate": "outputs/reports/chip_m64_seed21_ref400_dev32_v1.json",
        "control": "outputs/reports/chip_m64_control_seed21_ref400_dev32_v1.json",
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
        default=ROOT / "outputs/reports/chip_m64_inference_seed_certification_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def load(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def case_rows(path: str | Path) -> list[dict]:
    path = ROOT / path
    cases_path = path.with_name(f"{path.stem}_cases.json")
    return json.loads(cases_path.read_text(encoding="utf-8"))[KEY]


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
    return {
        "candidate": config["candidate"],
        "control": config["control"],
        "candidate_absolute": candidate,
        "control_absolute": control,
        "relative": relative,
        "paired": paired,
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
        for seed, config in REPORTS.items()
    }
    primary = {
        metric: {
            "mean_relative": sum(
                result["relative"][metric] for result in seeds.values()
            )
            / len(seeds),
            "improved_seeds": sum(
                result["relative"][metric] < 0.0 for result in seeds.values()
            ),
            "worsened_seeds": sum(
                result["relative"][metric] > 0.0 for result in seeds.values()
            ),
        }
        for metric in ("legalized_hard_cost", "legalized_v_rel")
    }
    checks = {
        "all_feasible": all(result["feasibility_both_one"] for result in seeds.values()),
        "joint_primary_in_at_least_two_seeds": sum(
            result["joint_primary_improves"] for result in seeds.values()
        )
        >= 2,
        "both_primary_mean_relative_negative": all(
            value["mean_relative"] < 0.0 for value in primary.values()
        ),
    }
    report = {
        "version": "chip-m64-inference-seed-certification-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "protocol": {
            "checkpoint_state": "ema",
            "nfe": 4,
            "refiner_steps": 400,
            "cases": 32,
            "legalizer": "scale_pack",
        },
        "checks": checks,
        "primary_summary": primary,
        "seeds": seeds,
        "claim_boundary": (
            "This tests inference-seed stability only. Training-seed and external "
            "generalization remain required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
