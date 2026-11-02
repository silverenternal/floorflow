#!/usr/bin/env python3
"""Compare M64 and its matched control under identical confirmation budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floorflow.evaluation.paired_bootstrap import paired_delta

DEFAULT_CANDIDATE_TRAIN = (
    ROOT
    / "outputs/chip_hierarchy/m64_confirmation_5k_v1/candidate/"
    "m64_candidate_confirmation5k.json"
)
DEFAULT_CONTROL_TRAIN = (
    ROOT
    / "outputs/chip_hierarchy/m64_confirmation_5k_v1/control/"
    "m64_control_confirmation5k.json"
)

PRIMARY_METRICS = ("legalized_hard_cost", "legalized_v_rel")
STRUCTURAL_METRICS = (
    "legalized_area_gap",
    "legalized_b2b_hpwl_gap",
    "legalized_p2b_hpwl_gap",
    "legalized_grouping",
    "legalized_boundary",
    "legalized_rho_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-train", type=Path, default=DEFAULT_CANDIDATE_TRAIN)
    parser.add_argument("--control-train", type=Path, default=DEFAULT_CONTROL_TRAIN)
    parser.add_argument("--candidate-ref400", type=Path, required=True)
    parser.add_argument("--control-ref400", type=Path, required=True)
    parser.add_argument("--candidate-ref1600", type=Path, default=None)
    parser.add_argument("--control-ref1600", type=Path, default=None)
    parser.add_argument("--ref400-key", default="nfe4_refine400")
    parser.add_argument("--ref1600-key", default="nfe4_refine1600")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=19)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_m64_confirmation_decision_v1.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_cases.json")


def relative(candidate: dict[str, Any], control: dict[str, Any], keys: tuple[str, ...]) -> dict:
    return {
        key: {
            "control": float(control[key]),
            "candidate": float(candidate[key]),
            "relative": float(candidate[key]) / float(control[key]) - 1.0,
        }
        for key in keys
    }


def paired_metrics(
    candidate_cases: list[dict],
    control_cases: list[dict],
    metrics: tuple[str, ...],
    *,
    iterations: int,
    seed: int,
) -> dict:
    return {
        metric: paired_delta(
            candidate_cases,
            control_cases,
            metric,
            iterations=iterations,
            seed=seed,
            lower_is_better=True,
        )
        for metric in metrics
    }


def budget_checks(candidate_train: dict, control_train: dict) -> dict:
    candidate_budget = candidate_train["execution_budget"]
    control_budget = control_train["execution_budget"]
    candidate_config = candidate_train["config"]
    control_config = control_train["config"]
    return {
        "same_steps": candidate_budget["planned_steps"] == control_budget["planned_steps"],
        "same_effective_batch": (
            candidate_budget["effective_batch"] == control_budget["effective_batch"]
        ),
        "same_training_seed": candidate_config["seed"] == control_config["seed"],
        "same_data_seed": candidate_config["data_seed"] == control_config["data_seed"],
        "same_parameter_count": (
            candidate_train["model_parameters"] == control_train["model_parameters"]
        ),
        "same_train_subset": candidate_train["train_cases"] == control_train["train_cases"],
    }


def stage_summary(
    candidate_report: dict,
    control_report: dict,
    key: str,
    *,
    iterations: int,
    seed: int,
) -> dict:
    candidate = candidate_report["results"][key]
    control = control_report["results"][key]
    metrics = (*PRIMARY_METRICS, *STRUCTURAL_METRICS)
    candidate_cases = read_json(cases_path(Path(candidate_report["_path"])))[key]
    control_cases = read_json(cases_path(Path(control_report["_path"])))[key]
    return {
        "key": key,
        "cases": int(candidate["cases"]),
        "candidate_absolute": candidate,
        "control_absolute": control,
        "relative": relative(candidate, control, metrics),
        "paired": paired_metrics(
            candidate_cases,
            control_cases,
            (*PRIMARY_METRICS, *STRUCTURAL_METRICS),
            iterations=iterations,
            seed=seed,
        ),
        "feasibility_both_one": (
            float(candidate["legalized_feasible_fraction"]) == 1.0
            and float(control["legalized_feasible_fraction"]) == 1.0
        ),
    }


def report_stage(path: Path, key: str) -> dict:
    report = read_json(path)
    if key not in report.get("results", {}):
        raise KeyError(f"{key} absent from {path}")
    report["_path"] = str(path)
    return report


def main() -> None:
    args = parse_args()
    candidate_train = read_json(args.candidate_train)
    control_train = read_json(args.control_train)
    candidate_ref400 = report_stage(args.candidate_ref400, args.ref400_key)
    control_ref400 = report_stage(args.control_ref400, args.ref400_key)
    ref400 = stage_summary(
        candidate_ref400,
        control_ref400,
        args.ref400_key,
        iterations=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    checks = budget_checks(candidate_train, control_train)
    checks.update(
        {
            "ref400_feasibility_both_one": ref400["feasibility_both_one"],
            "ref400_cost_nonregressing": ref400["relative"]["legalized_hard_cost"]["relative"] <= 0.0,
            "ref400_v_rel_nonregressing": ref400["relative"]["legalized_v_rel"]["relative"] <= 0.0,
            "ref400_p2b_not_catastrophic": (
                ref400["relative"]["legalized_p2b_hpwl_gap"]["relative"] <= 0.2
            ),
            "ref400_grouping_not_regressing": (
                ref400["relative"]["legalized_grouping"]["relative"] <= 0.0
            ),
        }
    )
    ref1600 = None
    if args.candidate_ref1600 is not None and args.control_ref1600 is not None:
        candidate_ref1600 = report_stage(args.candidate_ref1600, args.ref1600_key)
        control_ref1600 = report_stage(args.control_ref1600, args.ref1600_key)
        ref1600 = stage_summary(
            candidate_ref1600,
            control_ref1600,
            args.ref1600_key,
            iterations=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        checks.update(
            {
                "ref1600_feasibility_both_one": ref1600["feasibility_both_one"],
                "ref1600_cost_nonregressing": (
                    ref1600["relative"]["legalized_hard_cost"]["relative"] <= 0.0
                ),
                "ref1600_v_rel_nonregressing": (
                    ref1600["relative"]["legalized_v_rel"]["relative"] <= 0.0
                ),
            }
        )
    if not all(checks.values()):
        status = "fail_confirmation"
    elif ref1600 is None:
        status = "pending_ref1600"
    else:
        status = "pass_confirmation"
    decision = {
        "version": "chip-m64-confirmation-decision-v1",
        "status": status,
        "candidate_train": str(args.candidate_train),
        "control_train": str(args.control_train),
        "checks": checks,
        "formation": {
            "candidate_parameters": candidate_train["model_parameters"],
            "control_parameters": control_train["model_parameters"],
            "candidate_budget": candidate_train["execution_budget"],
            "control_budget": control_train["execution_budget"],
            "candidate_raw_final": candidate_train["final_dev"],
            "control_raw_final": control_train["final_dev"],
        },
        "ref400": ref400,
        "ref1600": ref1600,
        "sota_ready": False,
        "claim_boundary": (
            "A pass selects M64 for independently seeded and named-budget evaluation. "
            "It does not establish Trinity superiority or routed-PPA improvement."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
