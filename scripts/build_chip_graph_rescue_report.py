#!/usr/bin/env python3
"""Freeze the matched graph-conditioned global-slot rescue result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=ROOT / "outputs/chip_graph_rescue/screen_5k",
    )
    parser.add_argument("--control-arm", default="global_matched198_seed19")
    parser.add_argument("--candidate-arm", default="graph_slot_seed19")
    parser.add_argument("--scope", default="seed-19 5k matched graph-conditioning rescue screen")
    parser.add_argument("--decision", default="extend graph-slot to 20k and three seeds")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_graph_conditioned_rescue_v1.json",
    )
    return parser.parse_args()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(
    control_rows: list[dict],
    candidate_rows: list[dict],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict:
    result = paired_delta(
        candidate_rows,
        control_rows,
        metric,
        iterations=iterations,
        seed=seed,
        lower_is_better=True,
    )
    return {
        "cases": result["cases"],
        "control_mean": result["baseline_mean"],
        "candidate_mean": result["baseline_mean"] + result["mean_delta"],
        "relative_delta": result["relative_delta"],
        "paired_delta": result["mean_delta"],
        "ci95": [result["ci_low"], result["ci_high"]],
    }


def stratified(
    control_rows: list[dict],
    candidate_rows: list[dict],
) -> dict:
    control = {int(row["index"]): row for row in control_rows}
    candidate = {int(row["index"]): row for row in candidate_rows}
    result = {}
    for name, indices in (
        ("21-60", range(40)),
        ("61-120", range(40, 100)),
    ):
        row = {}
        for metric in ("legalized_hard_cost", "legalized_v_rel"):
            left = float(np.mean([control[index][metric] for index in indices]))
            right = float(np.mean([candidate[index][metric] for index in indices]))
            row[metric] = {
                "control": left,
                "graph": right,
                "relative_delta": (right - left) / left,
            }
        result[name] = row
    return result


def main() -> None:
    args = parse_args()
    root = args.screen_root
    control = root / args.control_arm
    candidate = root / args.candidate_arm
    control_report = read(root / f"{args.control_arm}.json")
    graph_report = read(root / f"{args.candidate_arm}.json")
    official = {}
    refined = {}
    strata = {}
    for name, path in (
        ("raw", "official_test100_nfe8_legalized_cases.json"),
        ("refined", "refiner_nfe4_ref400_cases.json"),
    ):
        left = read(control / path)
        right = read(candidate / path)
        if name == "refined":
            left = left["nfe4_refine400"]
            right = right["nfe4_refine400"]
        target = official if name == "raw" else refined
        strata[name] = stratified(left, right)
        target["legalized_hard_cost"] = paired(
            left,
            right,
            "legalized_hard_cost",
            iterations=args.bootstrap,
            seed=20261028,
        )
        target["legalized_v_rel"] = paired(
            left,
            right,
            "legalized_v_rel",
            iterations=args.bootstrap,
            seed=20261029,
        )
    external = read(ROOT / "outputs/reports/trinity_d192_matched_protocol_v2.json")
    graph_refined = read(candidate / "refiner_nfe4_ref400.json")["results"][
        "nfe4_refine400"
    ]
    parameter_error = abs(
        graph_report["model_parameters"] - control_report["model_parameters"]
    ) / control_report["model_parameters"]
    checks = {
        "parameter_match_within_1_percent": parameter_error <= 0.01,
        "dev_improves": (
            graph_report["final_dev"]["total"]
            < control_report["final_dev"]["total"]
        ),
        "official_both_metrics_improve": all(
            official[metric]["relative_delta"] < 0.0
            and official[metric]["ci95"][1] < 0.0
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
        "refined_both_metrics_improve": all(
            refined[metric]["relative_delta"] < 0.0
            and refined[metric]["ci95"][1] < 0.0
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": args.scope,
        "checks": checks,
        "parameter_match": {
            "control": control_report["model_parameters"],
            "graph": graph_report["model_parameters"],
            "relative_error": parameter_error,
        },
        "dev": {
            "control_total": control_report["final_dev"]["total"],
            "graph_total": graph_report["final_dev"]["total"],
            "control_latent_mse": control_report["final_dev"]["latent_mse"],
            "graph_latent_mse": graph_report["final_dev"]["latent_mse"],
        },
        "official": official,
        "refined": refined,
        "by_block": strata,
        "external_same_protocol": {
            "source": "outputs/reports/trinity_d192_matched_protocol_v2.json",
            "trinity_hard_cost": external["mean_cost"],
            "trinity_v_rel": external["mean_v_rel"],
            "graph_refined_hard_cost": graph_refined["legalized_hard_cost"],
            "graph_refined_v_rel": graph_refined["legalized_v_rel"],
            "hard_cost_relative_gap": (
                graph_refined["legalized_hard_cost"] / external["mean_cost"] - 1.0
            ),
        },
        "decision": args.decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
