#!/usr/bin/env python3
"""Summarize paired CubiCasa transfer tests across seeds and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def arm_name(base: str, seed: int) -> str:
    return base if seed == 19 else f"{base}_s{seed}"


def load_cases(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["index"]): row for row in rows}


def paired_delta(
    root: Path,
    scratch: str,
    candidate: str,
    *,
    seed: int,
) -> dict:
    scratch_path = root / arm_name(scratch, seed) / "metrics_test_cases.json"
    candidate_path = root / arm_name(candidate, seed) / "metrics_test_cases.json"
    left = load_cases(scratch_path)
    right = load_cases(candidate_path)
    indices = sorted(set(left) & set(right))
    rng = np.random.default_rng(20261026 + seed)
    result = {"seed": seed, "cases": len(indices)}
    for metric in ("bbox_iou", "center_l1", "size_l1"):
        delta = np.asarray(
            [right[index][metric] - left[index][metric] for index in indices]
        )
        bootstrap = np.asarray(
            [rng.choice(delta, len(delta), replace=True).mean() for _ in range(4000)]
        )
        result[metric] = {
            "scratch": float(np.mean([left[index][metric] for index in indices])),
            "candidate": float(np.mean([right[index][metric] for index in indices])),
            "delta": float(delta.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    outside = np.asarray(
        [int(right[index]["outside"]) - int(left[index]["outside"]) for index in indices]
    )
    result["outside_fraction"] = {
        "scratch": float(np.mean([left[index]["outside"] for index in indices])),
        "candidate": float(np.mean([right[index]["outside"] for index in indices])),
        "delta": float(outside.mean()),
    }
    return result


def main() -> None:
    args = parse_args()
    full_finetune = [
        paired_delta(
            args.root,
            "house_scratch",
            "house_chip_init",
            seed=seed,
        )
        for seed in (19, 20, 21)
    ]
    huber = [
        paired_delta(
            args.root,
            "house_scratch",
            "house_huber",
            seed=seed,
        )
        for seed in (19, 20, 21)
    ]
    huber_deltas = [row["bbox_iou"]["delta"] for row in huber]
    huber_relative = [
        row["bbox_iou"]["delta"] / row["bbox_iou"]["scratch"] for row in huber
    ]
    deltas = [row["bbox_iou"]["delta"] for row in full_finetune]
    relative = [
        row["bbox_iou"]["delta"] / row["bbox_iou"]["scratch"]
        for row in full_finetune
    ]
    summary = {
        "status": "complete",
        "protocol": {
            "task": "CubiCasa room bounding-box generation",
            "train": 4184,
            "val": 400,
            "test": 398,
            "downstream_steps": 5000,
            "sampling_steps": 8,
            "seeds": [19, 20, 21],
        },
        "full_finetune": full_finetune,
        "full_finetune_seed_summary": {
            "bbox_iou_delta_mean": float(np.mean(deltas)),
            "bbox_iou_delta_std": float(np.std(deltas, ddof=1)),
            "relative_delta_mean": float(np.mean(relative)),
            "positive_seeds": sum(value > 0 for value in deltas),
        },
        "frozen8": [
            paired_delta(
                args.root,
                "house_frozen8_scratch",
                "house_frozen8_chip",
                seed=19,
            )
        ],
        "huber_flow": huber,
        "huber_seed_summary": {
            "bbox_iou_delta_mean": float(np.mean(huber_deltas)),
            "bbox_iou_delta_std": float(np.std(huber_deltas, ddof=1)),
            "relative_delta_mean": float(np.mean(huber_relative)),
            "positive_seeds": sum(value > 0 for value in huber_deltas),
        },
        "gate": {
            "status": (
                "pass"
                if all(value > 0 for value in deltas) and np.mean(relative) >= 0.10
                else "fail"
            ),
            "rule": "all three seeds positive on paired test bbox IoU and mean relative gain >= 10%",
        },
        "huber_gate": {
            "status": (
                "pass"
                if all(value > 0 for value in huber_deltas)
                and np.mean(huber_relative) >= 0.10
                else "fail"
            ),
            "rule": "all three seeds positive on paired test bbox IoU and mean relative gain >= 10%",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["full_finetune_seed_summary"], indent=2))
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
