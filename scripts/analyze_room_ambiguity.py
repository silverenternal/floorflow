#!/usr/bin/env python3
"""Estimate contour-conditioned layout ambiguity and a room IoU ceiling proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.core import room_area_distribution


def _descriptor(footprint: np.ndarray) -> np.ndarray:
    values = footprint.astype(np.float32)
    stride = max(1, values.shape[0] // 16)
    coarse = values[::stride, ::stride][:16, :16]
    if coarse.shape != (16, 16):
        coarse = np.pad(coarse, ((0, 16 - coarse.shape[0]), (0, 16 - coarse.shape[1])))
    return np.concatenate([coarse.ravel(), [values.mean(), values.sum() / values.size]])


def _entropy(distribution: np.ndarray) -> float:
    probabilities = distribution[distribution > 0.0]
    return float(-(probabilities * np.log(probabilities)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/room_ambiguity/condition_entropy.json")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--test-limit", type=int, default=300)
    parser.add_argument("--neighbors", type=int, default=12)
    args = parser.parse_args()
    train = CubiCasaFloorplanDataset(
        args.data_root, "train", image_size=128, cache_root=args.cache_root, limit=args.train_limit
    )
    test = CubiCasaFloorplanDataset(
        args.data_root, "test", image_size=128, cache_root=args.cache_root, limit=args.test_limit
    )
    train_features = np.stack([_descriptor(train[index][1].numpy()) for index in range(len(train))])
    train_distributions = np.stack(
        [room_area_distribution(train[index][0].numpy(), train.taxonomy) for index in range(len(train))]
    )
    train_norm = np.linalg.norm(train_features, axis=1, keepdims=True).clip(min=1.0e-6)
    normalized_train = train_features / train_norm
    rows = []
    for index in range(len(test)):
        target, footprint, _floor, name = test[index]
        descriptor = _descriptor(footprint.numpy())
        scores = normalized_train @ (descriptor / max(np.linalg.norm(descriptor), 1.0e-6))
        nearest = np.argpartition(-scores, kth=min(args.neighbors, len(scores) - 1))[: args.neighbors]
        neighborhood_mean = train_distributions[nearest].mean(axis=0)
        expected = room_area_distribution(target.numpy(), test.taxonomy)
        rows.append(
            {
                "case": name,
                "entropy": _entropy(neighborhood_mean),
                "neighborhood_area_mae": float(np.abs(neighborhood_mean - expected).mean()),
                "target_entropy": _entropy(expected),
            }
        )
    report = {
        "train_samples": len(train),
        "test_samples": len(test),
        "neighbors": args.neighbors,
        "mean_condition_entropy": float(np.mean([row["entropy"] for row in rows])),
        "mean_neighborhood_area_mae": float(np.mean([row["neighborhood_area_mae"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("train_samples", "test_samples", "mean_condition_entropy", "mean_neighborhood_area_mae")}, indent=2))


if __name__ == "__main__":
    main()
