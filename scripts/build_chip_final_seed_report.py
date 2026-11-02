#!/usr/bin/env python3
"""Build the frozen-runtime three-seed graph-slot promotion report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (19, 20, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs/chip_final/screen_20k",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/chip_final_three_seed_evidence_v1.json",
    )
    return parser.parse_args()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path, *, refined: bool) -> list[dict]:
    payload = read(path)
    return payload["nfe4_refine400"] if refined else payload


def summary(path: Path, *, refined: bool) -> dict:
    payload = read(path)
    if refined:
        payload = payload["results"]["nfe4_refine400"]
    return {
        "legalized_hard_cost": float(payload["legalized_hard_cost"]),
        "legalized_v_rel": float(payload["legalized_v_rel"]),
        "legalized_feasible_fraction": float(
            payload["legalized_feasible_fraction"]
        ),
    }


def paired_rows(
    control: list[dict],
    candidate: list[dict],
    *,
    iterations: int,
    seed: int,
) -> dict:
    output = {}
    for metric in ("legalized_hard_cost", "legalized_v_rel"):
        output[metric] = paired_delta(
            candidate,
            control,
            metric,
            iterations=iterations,
            seed=seed,
            lower_is_better=True,
        )
    return output


def arm_names(seed: int) -> tuple[str, str]:
    return (
        f"graph_velocity_seed{seed}_20k_final",
        f"global_matched198_seed{seed}_20k_final",
    )


def main() -> None:
    args = parse_args()
    results = {}
    for seed in SEEDS:
        graph_arm, control_arm = arm_names(seed)
        graph_dir = args.root / graph_arm
        control_dir = args.root / control_arm
        graph_training = read(args.root / f"{graph_arm}.json")
        control_training = read(args.root / f"{control_arm}.json")
        entry = {
            "graph_parameters": int(graph_training["model_parameters"]),
            "control_parameters": int(control_training["model_parameters"]),
            "graph_dev_total": float(graph_training["final_dev"]["total"]),
            "control_dev_total": float(control_training["final_dev"]["total"]),
            "raw": {
                "graph": summary(
                    graph_dir / "official_test100_nfe8_legalized.json",
                    refined=False,
                ),
                "control": summary(
                    control_dir / "official_test100_nfe8_legalized.json",
                    refined=False,
                ),
            },
            "raw_ema": {
                "graph": summary(
                    graph_dir / "official_test100_nfe8_legalized_ema.json",
                    refined=False,
                ),
                "control": summary(
                    control_dir / "official_test100_nfe8_legalized_ema.json",
                    refined=False,
                ),
            },
            "refined": {
                "graph": summary(
                    graph_dir / "refiner_nfe4_ref400.json",
                    refined=True,
                ),
                "control": summary(
                    control_dir / "refiner_nfe4_ref400.json",
                    refined=True,
                ),
            },
        }
        for kind, graph_file, control_file in (
            (
                "raw",
                "official_test100_nfe8_legalized_cases.json",
                "official_test100_nfe8_legalized_cases.json",
            ),
            (
                "refined",
                "refiner_nfe4_ref400_cases.json",
                "refiner_nfe4_ref400_cases.json",
            ),
        ):
            entry[f"{kind}_paired"] = paired_rows(
                rows(control_dir / control_file, refined=kind == "refined"),
                rows(graph_dir / graph_file, refined=kind == "refined"),
                iterations=args.bootstrap,
                seed=20261100 + seed + (100 if kind == "refined" else 0),
            )
        entry["passes_both_metrics"] = all(
            entry["refined_paired"][metric]["mean_delta"] < 0.0
            and entry["refined_paired"][metric]["ci_high"] < 0.0
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        )
        results[str(seed)] = entry

    means = {
        kind: {
            metric: float(
                np.mean(
                    [
                        results[str(seed)][f"{kind}_paired"][metric][
                            "relative_delta"
                        ]
                        for seed in SEEDS
                    ]
                )
            )
            for metric in ("legalized_hard_cost", "legalized_v_rel")
        }
        for kind in ("raw", "refined")
    }
    absolute_means = {
        kind: {
            arm: {
                metric: float(
                    np.mean(
                        [
                            results[str(seed)][kind][arm][metric]
                            for seed in SEEDS
                        ]
                    )
                )
                for metric in (
                    "legalized_hard_cost",
                    "legalized_v_rel",
                    "legalized_feasible_fraction",
                )
            }
            for arm in ("graph", "control")
        }
        for kind in ("raw", "raw_ema", "refined")
    }
    checks = {
        "all_seed_reports_complete": all(
            entry["refined_paired"]["legalized_hard_cost"]["cases"] == 100
            for entry in results.values()
        ),
        "graph_wins_both_metrics_all_seeds": all(
            entry["passes_both_metrics"] for entry in results.values()
        ),
        "refined_seed_mean_cost_improves": (
            means["refined"]["legalized_hard_cost"] < 0.0
        ),
        "refined_seed_mean_v_rel_improves": (
            means["refined"]["legalized_v_rel"] < 0.0
        ),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "frozen-runtime three-seed graph-slot versus parameter-matched global control",
        "protocol": {
            "steps": 20000,
            "batch_size": 128,
            "raw": "NFE8, scale_pack, score_full_fast",
            "refined": "NFE4, 400-step closed-form refiner, scale_pack",
            "best_of_n_used": False,
            "target_derived_control": False,
        },
        "checks": checks,
        "seed_mean_relative_delta": means,
        "seed_mean_absolute": absolute_means,
        "seeds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **checks}, indent=2))


if __name__ == "__main__":
    main()
