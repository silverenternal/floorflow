#!/usr/bin/env python3
"""Summarize paired direct-token and global-slot results across three seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/plan_c_global_slot_seeds_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(path: Path) -> dict[int, dict]:
    return {int(row["index"]): row for row in read_json(path)}


def paired_metric(
    control_path: Path,
    candidate_path: Path,
    metric: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    control = load_cases(control_path)
    candidate = load_cases(candidate_path)
    indices = sorted(set(control) & set(candidate))
    delta = np.asarray(
        [
            float(candidate[index][metric]) - float(control[index][metric])
            for index in indices
        ]
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(delta, len(delta), replace=True).mean()
            for _ in range(bootstrap)
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
        "relative_delta": (
            (candidate_mean - control_mean) / control_mean
            if control_mean != 0
            else 0.0
        ),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def path_map(seed: int) -> dict[str, Path]:
    if seed == 19:
        return {
            "baseline_report": REPO_ROOT
            / "outputs/chip_large/v2_20k/dense_no_ot_huber.json",
            "global_report": REPO_ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10_20k.json",
            "global_early_report": REPO_ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10.json",
            "baseline_cases": REPO_ROOT
            / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized_cases.json",
            "baseline_summary": REPO_ROOT
            / "outputs/chip_large/v2_20k/dense_no_ot_huber/official_test100_nfe8_legalized.json",
            "global_cases": REPO_ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized_cases.json",
            "global_summary": REPO_ROOT
            / "outputs/chip_slot/global_v1/global_slot_overlap10_20k/official_test100_nfe8_legalized.json",
        }
    suffix = f"_s{seed}"
    return {
        "baseline_report": REPO_ROOT
        / f"outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber{suffix}.json",
        "global_report": REPO_ROOT
        / f"outputs/chip_slot/global_v2/global_slot_seed{seed}_overlap10_20k.json",
        "global_early_report": REPO_ROOT
        / f"outputs/chip_slot/global_v2/global_slot_seed{seed}_overlap10.json",
        "baseline_cases": REPO_ROOT
        / f"outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber{suffix}/official_test100_nfe8_legalized_cases.json",
        "baseline_summary": REPO_ROOT
        / f"outputs/chip_large/v2_20k_seed_replication/dense_no_ot_huber{suffix}/official_test100_nfe8_legalized.json",
        "global_cases": REPO_ROOT
        / f"outputs/chip_slot/global_v2/global_slot_seed{seed}_overlap10_20k/official_test100_nfe8_legalized_cases.json",
        "global_summary": REPO_ROOT
        / f"outputs/chip_slot/global_v2/global_slot_seed{seed}_overlap10_20k/official_test100_nfe8_legalized.json",
    }


def report_final(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing"}
    report = read_json(path)
    history = report.get("history", [])
    return {
        "status": report.get("status"),
        "path": str(path),
        "final_dev": report.get("final_dev"),
        "history": history,
        "training_seconds": history[-1].get("elapsed_seconds")
        if history
        else None,
    }


def official_summary(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    report = read_json(path)
    return {
        key: report.get(key)
        for key in (
            "cases",
            "raw_hard_cost",
            "raw_feasible_fraction",
            "legalized_hard_cost",
            "legalized_feasible_fraction",
            "legalized_v_rel",
            "legalized_hpwl_gap",
            "legalized_area_gap",
            "overlap",
            "mib",
            "aspect_diversity_ratio",
        )
    } | {"status": "complete", "path": str(path)}


def first_total_step(report: dict, threshold: float) -> int | None:
    for row in report.get("history", []):
        if float(row.get("total", float("inf"))) <= threshold:
            return int(row["step"])
    return None


def length_bands(
    control_path: Path,
    candidate_path: Path,
    metric: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    control = load_cases(control_path)
    candidate = load_cases(candidate_path)
    bands = {"21-60": (21, 60), "61-120": (61, 120)}
    result = {}
    for offset, (name, (low, high)) in enumerate(bands.items()):
        indices = sorted(
            index
            for index in set(control) & set(candidate)
            if low <= int(control[index]["blocks"]) <= high
        )
        delta = np.asarray(
            [
                float(candidate[index][metric]) - float(control[index][metric])
                for index in indices
            ]
        )
        rng = np.random.default_rng(seed + offset)
        draws = np.asarray(
            [
                rng.choice(delta, len(delta), replace=True).mean()
                for _ in range(bootstrap)
            ]
        )
        result[name] = {
            "cases": len(indices),
            "paired_delta": float(delta.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = [19, 20, 21]
    rows = []
    for seed in seeds:
        paths = path_map(seed)
        baseline = report_final(paths["baseline_report"])
        candidate = report_final(paths["global_report"])
        if paths["global_early_report"].is_file():
            early = report_final(paths["global_early_report"])
            candidate["history"] = early.get("history", []) + candidate.get(
                "history", []
            )
        row = {
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "official_baseline": official_summary(paths["baseline_summary"]),
            "official_candidate": official_summary(paths["global_summary"]),
        }
        if (
            paths["baseline_cases"].is_file()
            and paths["global_cases"].is_file()
        ):
            row["official_paired"] = {
                metric: paired_metric(
                    paths["baseline_cases"],
                    paths["global_cases"],
                    metric,
                    bootstrap=args.bootstrap,
                    seed=20261026 + seed + offset,
                )
                for offset, metric in enumerate(
                    (
                        "legalized_hard_cost",
                        "legalized_v_rel",
                        "legalized_hpwl_gap",
                        "legalized_area_gap",
                    )
                )
            }
            row["official_by_length"] = {
                metric: length_bands(
                    paths["baseline_cases"],
                    paths["global_cases"],
                    metric,
                    bootstrap=max(1000, args.bootstrap // 2),
                    seed=20262026 + seed + offset,
                )
                for offset, metric in enumerate(
                    ("legalized_hard_cost", "legalized_v_rel")
                )
            }
        else:
            row["official_paired"] = {"status": "missing"}
            row["official_by_length"] = {"status": "missing"}
        rows.append(row)

    official_rows = [
        row for row in rows if isinstance(row["official_paired"], dict)
        and "legalized_hard_cost" in row["official_paired"]
    ]
    hard_deltas = [
        row["official_paired"]["legalized_hard_cost"]["relative_delta"]
        for row in official_rows
    ]
    vrel_deltas = [
        row["official_paired"]["legalized_v_rel"]["relative_delta"]
        for row in official_rows
    ]
    gate_pass = (
        len(official_rows) == len(seeds)
        and all(value < 0 for value in hard_deltas)
        and all(value < 0 for value in vrel_deltas)
        and float(np.mean(hard_deltas)) <= -0.10
    )
    threshold_steps = []
    for row in rows:
        baseline_step = first_total_step(row["baseline"], 0.04)
        candidate_step = first_total_step(row["candidate"], 0.04)
        threshold_steps.append(
            {
                "seed": row["seed"],
                "baseline_step": baseline_step,
                "candidate_step": candidate_step,
                "step_reduction": (
                    (baseline_step - candidate_step) / baseline_step
                    if baseline_step is not None and candidate_step is not None
                    else None
                ),
            }
        )
    step_reductions = [
        row["step_reduction"]
        for row in threshold_steps
        if row["step_reduction"] is not None
    ]
    report = {
        "status": "complete" if len(official_rows) == len(seeds) else "incomplete",
        "protocol": {
            "seeds": seeds,
            "official_cases": 100,
            "sampling_steps": 8,
            "primary_metrics": [
                "legalized_hard_cost",
                "legalized_v_rel",
            ],
            "best_of_n_used": False,
        },
        "main_seed19_schedule": (
            "2k base, 5k resume with overlap scale 10, "
            "20k resume with overlap scale 10"
        ),
        "seeds": rows,
        "official_seed_summary": {
            "complete_seeds": len(official_rows),
            "hard_cost_relative_deltas": hard_deltas,
            "hard_cost_mean_relative_delta": (
                float(np.mean(hard_deltas)) if hard_deltas else None
            ),
            "vrel_relative_deltas": vrel_deltas,
            "vrel_mean_relative_delta": (
                float(np.mean(vrel_deltas)) if vrel_deltas else None
            ),
            "hard_cost_improved_seeds": sum(value < 0 for value in hard_deltas),
            "vrel_improved_seeds": sum(value < 0 for value in vrel_deltas),
            "mean_step_reduction_to_total_0_04": (
                float(np.mean(step_reductions)) if step_reductions else None
            ),
        },
        "convergence": {
            "total_threshold": 0.04,
            "seeds": threshold_steps,
        },
        "gate": {
            "status": "pass" if gate_pass else "incomplete-or-fail",
            "rule": (
                "All three seeds lower official hard cost and V_rel, "
                "with mean hard-cost improvement >= 10%."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["official_seed_summary"], indent=2))
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
