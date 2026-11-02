#!/usr/bin/env python3
"""Select a Raster2Seq EMA blend on validation with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.statistical_effect_sizes import (
    cliffs_delta,
    rank_biserial_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("room_f1", "room_iou", "corner_f1", "room_sem_f1", "window_door_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/"
        "official_eval_val503_v1/result_jsons",
    )
    parser.add_argument(
        "--blend-root",
        type=Path,
        default=ROOT
        / "outputs/external_baselines/raster2seq/checkpoint_blends_v1",
    )
    parser.add_argument(
        "--result-template",
        default=(
            "outputs/external_baselines/raster2seq/"
            "official_eval_val503_blend_{alpha}_v1/result_jsons"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster2seq_blend_selection_val_v1.json",
    )
    return parser.parse_args()


def read_cases(directory: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def main() -> None:
    args = parse_args()
    baseline = read_cases(args.baseline)
    manifest = json.loads(
        (args.blend_root / "manifest.json").read_text(encoding="utf-8")
    )
    candidates = {}
    for row in manifest["blends"]:
        alpha = f"{float(row['alpha']):.2f}"
        directory = ROOT / args.result_template.format(alpha=alpha)
        candidate = read_cases(directory)
        if set(candidate) != set(baseline):
            raise ValueError(f"case mismatch for alpha {alpha}")
        names = sorted(baseline)
        comparisons = {}
        for metric_index, metric in enumerate(METRICS):
            left = [{"value": candidate[name][metric]} for name in names]
            right = [{"value": baseline[name][metric]} for name in names]
            differences = np.asarray(
                [float(a["value"]) - float(b["value"]) for a, b in zip(left, right, strict=True)]
            )
            stats = paired_delta(
                left,
                right,
                "value",
                iterations=4000,
                seed=260101 + metric_index,
            )
            comparisons[metric] = {
                **stats,
                "candidate_mean": float(np.mean([item["value"] for item in left])),
                "baseline_mean": float(np.mean([item["value"] for item in right])),
                "rank_biserial": float(rank_biserial_correlation(differences)),
                "cliffs_delta": float(
                    cliffs_delta(
                        [float(item["value"]) for item in left],
                        [float(item["value"]) for item in right],
                    )
                ),
            }
        candidates[alpha] = {
            "checkpoint": row["path"],
            "checkpoint_sha256": row["sha256"],
            "comparisons": comparisons,
        }
    eligible = {
        alpha: values
        for alpha, values in candidates.items()
        if values["comparisons"]["room_f1"]["ci_low"] > 0.0
        and values["comparisons"]["room_iou"]["mean_delta"] >= 0.0
        and values["comparisons"]["corner_f1"]["mean_delta"] >= 0.0
    }
    selected = max(
        eligible or candidates,
        key=lambda alpha: candidates[alpha]["comparisons"]["room_f1"]["mean_delta"],
    )
    report = {
        "status": "complete" if eligible else "no_significant_blend",
        "scope": "Raster2Seq val503 EMA/non-EMA checkpoint selection",
        "selection_rule": (
            "Highest RoomF1 mean gain among blends with positive paired "
            "RoomF1 95% CI and nonnegative RoomIoU/corner-F1 deltas."
        ),
        "baseline": str(args.baseline.relative_to(ROOT)),
        "candidates": candidates,
        "eligible": sorted(eligible),
        "selected_alpha": selected,
        "selected_checkpoint": candidates[selected]["checkpoint"],
        "selected_checkpoint_sha256": candidates[selected]["checkpoint_sha256"],
        "official_test509_used": False,
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_alpha": selected,
                "selected": candidates[selected]["comparisons"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "complete" else 1)


__all__ = ["main"]


if __name__ == "__main__":
    main()
