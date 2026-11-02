#!/usr/bin/env python3
"""Analyze global-slot house generation against matched direct-flow controls."""

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
        default=REPO_ROOT
        / "outputs/reports/plan_c_house_global_slot_seeds_v1.json",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    return parser.parse_args()


def arm(base: str, seed: int) -> str:
    return base if seed == 19 else f"{base}_s{seed}"


def load_cases(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in rows}


def paired(
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
    rng = np.random.default_rng(20261026 + seed)
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
        "control": control_mean,
        "candidate": candidate_mean,
        "delta": float(delta.mean()),
        "relative_delta": (
            (candidate_mean - control_mean) / control_mean
            if control_mean != 0
            else 0.0
        ),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    baseline_root = REPO_ROOT / "outputs/house_cross_domain/v1"
    candidate_root = REPO_ROOT / "outputs/house_cross_domain/v3_global_slot"
    rows = []
    for seed in (19, 20, 21):
        control_path = (
            baseline_root / arm("house_scratch", seed) / "metrics_test_cases.json"
        )
        candidate_path = (
            candidate_root
            / arm("house_global_slot", seed)
            / "metrics_test_cases.json"
        )
        if not control_path.is_file() or not candidate_path.is_file():
            rows.append(
                {
                    "seed": seed,
                    "status": "missing",
                    "control": str(control_path),
                    "candidate": str(candidate_path),
                }
            )
            continue
        row = {"seed": seed, "status": "complete"}
        for offset, metric in enumerate(("bbox_iou", "center_l1", "size_l1")):
            row[metric] = paired(
                control_path,
                candidate_path,
                metric,
                bootstrap=args.bootstrap,
                seed=seed + offset,
            )
        control = load_cases(control_path)
        candidate = load_cases(candidate_path)
        indices = sorted(set(control) & set(candidate))
        row["outside_fraction"] = {
            "control": float(
                np.mean([control[index]["outside"] for index in indices])
            ),
            "candidate": float(
                np.mean([candidate[index]["outside"] for index in indices])
            ),
        }
        rows.append(row)
    complete = [row for row in rows if row["status"] == "complete"]
    iou_deltas = [row["bbox_iou"]["delta"] for row in complete]
    relative = [row["bbox_iou"]["relative_delta"] for row in complete]
    gate_pass = (
        len(complete) == 3
        and all(value > 0 for value in iou_deltas)
        and float(np.mean(relative)) >= 0.10
    )
    report = {
        "status": "complete" if len(complete) == 3 else "incomplete",
        "protocol": {
            "task": "CubiCasa room bounding-box generation",
            "train": 4184,
            "val": 400,
            "test": 398,
            "training_steps": 5000,
            "sampling_steps": 8,
            "seeds": [19, 20, 21],
            "best_of_n_used": False,
            "control": "direct token flow trained from scratch",
            "candidate": "global latent flow plus slot decoder trained from scratch",
        },
        "seeds": rows,
        "summary": {
            "completed_seeds": len(complete),
            "bbox_iou_delta_mean": (
                float(np.mean(iou_deltas)) if iou_deltas else None
            ),
            "bbox_iou_relative_delta_mean": (
                float(np.mean(relative)) if relative else None
            ),
            "positive_seeds": sum(value > 0 for value in iou_deltas),
        },
        "gaussian_particle_controls": {
            name: {
                key: json.loads(path.read_text(encoding="utf-8")).get(key)
                for key in ("bbox_iou", "center_l1", "size_l1", "outside_fraction")
            }
            for name in ("house_gaussian_no_ot", "house_gaussian_minibatch_ot")
            for path in (
                REPO_ROOT
                / f"outputs/house_cross_domain/v2_gaussian_ot/{name}/metrics_test.json",
            )
            if path.is_file()
        },
        "gate": {
            "status": "pass" if gate_pass else "incomplete-or-fail",
            "rule": (
                "All three seeds improve paired bbox IoU and the mean "
                "relative improvement is at least 10%."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
