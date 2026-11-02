#!/usr/bin/env python3
"""Run the r156 train/validation size sweet-spot sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.evaluation.scaling_report import build_scaling_report

METRICS = ("room_macro_iou", "fpeval_wall_f1", "fpeval_opening_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        type=Path,
        default=REPO_ROOT / "outputs/train/distribution_retrieval_free_merged_style_v2.pt",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1",
    )
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--room-segmenter",
        type=Path,
        default=REPO_ROOT / "outputs/train/room_segmenter_crop_consistency_v1/best.pt",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 5126],
    )
    parser.add_argument(
        "--val-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 191],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/reports/scaling_sweet_spot_v1.json",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "outputs/train/scaling_r156",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def train_one(args: argparse.Namespace, size: int) -> Path:
    output = args.run_root / f"n{size}"
    _run(
        [
            sys.executable,
            "scripts/train_joint_distribution_cfm.py",
            "--initialize-distribution",
            str(args.initialize),
            "--output-dir",
            str(output),
            "--cache-root",
            str(args.cache_root),
            "--data-root",
            str(args.data_root),
            "--epochs",
            "1",
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--limit-train",
            str(size),
            "--limit-val",
            "32",
            "--minibatch-ot",
            "token",
            "--element-warmup-updates",
            "0",
            "--max-steps",
            str(args.steps),
            "--device",
            "cuda",
        ]
    )
    checkpoint = output / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"training did not produce {checkpoint}")
    return checkpoint


def evaluate_one(
    args: argparse.Namespace,
    checkpoint: Path,
    limit: int,
    output: Path,
    per_case: Path,
) -> tuple[dict, list[dict]]:
    _run(
        [
            sys.executable,
            "scripts/evaluate_distribution_overlay.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--per-case-output",
            str(per_case),
            "--split",
            "val",
            "--offset",
            "0",
            "--limit",
            str(limit),
            "--batch-size",
            "4",
            "--workers",
            "2",
            "--device",
            "cuda",
            "--room-segmenter-checkpoint",
            str(args.room_segmenter),
            "--room-prior-weight",
            "3.0",
            "--retrieval-prior-weight",
            "0.0",
            "--style-id",
            "0",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    rows = json.loads(per_case.read_text(encoding="utf-8"))["cases"]
    return report, rows


def composite(values: list[dict]) -> float:
    scores = [
        float(row[name])
        for row in values
        for name in METRICS
        if isinstance(row.get(name), (int, float))
    ]
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    train_rows = []
    checkpoints: dict[int, Path] = {}
    for size in args.train_sizes:
        checkpoint = train_one(args, size)
        checkpoints[size] = checkpoint
        report, rows = evaluate_one(
            args,
            checkpoint,
            64,
            args.run_root / f"n{size}_val64.json",
            args.run_root / f"n{size}_val64_cases.json",
        )
        summary = report["cad_metrics"]["summary"]
        train_rows.append(
            {
                "train_size": int(size),
                "val_size": 64,
                **{name: summary.get(name) for name in METRICS},
                "composite": composite(rows),
            }
        )
    best_size = max(checkpoints, key=lambda item: composite(
        json.loads(
            (args.run_root / f"n{item}_val64_cases.json").read_text(
                encoding="utf-8"
            )
        )["cases"]
    ))
    best_checkpoint = checkpoints[best_size]
    full_report, full_rows = evaluate_one(
        args,
        best_checkpoint,
        191,
        args.run_root / "best_val191.json",
        args.run_root / "best_val191_cases.json",
    )
    del full_report
    full_scores = [composite([row]) for row in full_rows]
    rng = np.random.default_rng(19)
    val_rows = []
    for size in args.val_sizes:
        report, rows = evaluate_one(
            args,
            best_checkpoint,
            int(size),
            args.run_root / f"val{size}.json",
            args.run_root / f"val{size}_cases.json",
        )
        summary = report["cad_metrics"]["summary"]
        subset_scores = [composite([row]) for row in rows]
        draws = [
            float(
                np.mean(
                    rng.choice(
                        full_scores,
                        size=int(size),
                        replace=False,
                    )
                )
            )
            for _ in range(200)
        ]
        bootstrap_std = float(np.std(draws))
        val_rows.append(
            {
                "val_size": int(size),
                **{name: summary.get(name) for name in METRICS},
                "composite": float(np.mean(subset_scores)),
                "metric_std": bootstrap_std,
                "rank_stability": float(1.0 / (1.0 + bootstrap_std)),
            }
        )
    report = build_scaling_report(train_rows, val_rows)
    report["best_train_size"] = best_size
    report["best_checkpoint"] = str(best_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["sweet_spot"], indent=2))


if __name__ == "__main__":
    main()
