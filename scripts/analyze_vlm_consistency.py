#!/usr/bin/env python3
"""Correlate VLM rubric scores with deterministic structural metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from floorflow.data.dataset import CubiCasaFloorplanDataset


def _case_key(name: str) -> str:
    return name.replace("__", "/", 1).removesuffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-report", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/cubicasa5k_full_v1"))
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    vlm = json.loads(args.vlm_report.read_text(encoding="utf-8"))
    rows = json.loads(args.per_case.read_text(encoding="utf-8"))
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "test",
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        write_cache=False,
    )
    names = [f"{sample.relative_case}__{sample.floor_id}" for sample in dataset.samples]
    metric_by_case = {name: metrics for name, metrics in zip(names, rows)}
    pairs = []
    for row in vlm["rows"]:
        if row.get("parse_error"):
            continue
        key = _case_key(Path(row["image"]).name)
        if key in metric_by_case:
            pairs.append((row, metric_by_case[key]))
    fields = {
        "room_plausibility": "room_area_distribution_mae",
        "topology_consistency": "opening_wall_alignment",
        "furniture_room_consistency": "furniture_room_containment",
        "drawing_readability": "room_area_coverage",
    }
    correlations = {}
    for score_field, metric_field in fields.items():
        scores = np.asarray([float(row.get(score_field, 0.0)) for row, _ in pairs])
        metrics = np.asarray([float(metrics.get(metric_field, 0.0)) for _, metrics in pairs])
        if len(scores) >= 3 and np.std(scores) > 0 and np.std(metrics) > 0:
            rho, p_value = spearmanr(scores, metrics)
        else:
            rho, p_value = 0.0, 1.0
        correlations[score_field] = {
            "metric": metric_field,
            "spearman_rho": float(rho),
            "p_value": float(p_value),
        }
    report = {
        "paired_samples": len(pairs),
        "correlations": correlations,
        "note": "This is objective consistency evidence, not a substitute for human inter-rater agreement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
