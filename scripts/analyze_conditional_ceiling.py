#!/usr/bin/env python3
"""Estimate contour-neighbour room IoU and layout ambiguity ceilings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from floorflow.evaluation.conditional_ceiling import (
    distribution_entropy,
    neighbour_iou_summary,
    room_macro_iou_numpy,
)
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.core import contour_descriptor, room_area_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.train_limit,
        write_cache=False,
    )
    query = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.query_limit,
        write_cache=False,
    )
    train_descriptors = []
    train_targets = []
    train_distributions = []
    for index in range(len(train)):
        target, footprint, _floor_id, _name = train[index]
        train_descriptors.append(contour_descriptor(footprint.numpy()))
        train_targets.append(target.numpy().astype(np.uint8))
        train_distributions.append(room_area_distribution(target.numpy(), train.taxonomy))
    descriptors = np.asarray(train_descriptors, dtype=np.float32)
    descriptor_norm = np.linalg.norm(descriptors, axis=1, keepdims=True).clip(min=1.0e-6)
    normalized = descriptors / descriptor_norm
    targets = np.stack(train_targets)
    distributions = np.stack(train_distributions)
    neighbor_count = min(max(1, args.neighbors), len(train))
    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(len(query)):
        target, footprint, _floor_id, name = query[index]
        descriptor = contour_descriptor(footprint.numpy())
        scores = normalized @ (descriptor / max(np.linalg.norm(descriptor), 1.0e-6))
        selected = np.argpartition(-scores, kth=neighbor_count - 1)[:neighbor_count]
        selected = selected[np.argsort(-scores[selected])]
        summary = neighbour_iou_summary(
            targets[selected],
            target.numpy(),
            query.taxonomy,
        )
        random_indices = rng.choice(len(train), size=neighbor_count, replace=False)
        random_iou = float(
            np.mean(
                [
                    room_macro_iou_numpy(
                        targets[random_index],
                        target.numpy(),
                        query.taxonomy,
                    )
                    for random_index in random_indices
                ]
            )
        )
        neighborhood_distribution = distributions[selected].mean(axis=0)
        rows.append(
            {
                "name": name,
                **summary,
                "random_neighbour_mean_iou": random_iou,
                "condition_entropy": distribution_entropy(
                    neighborhood_distribution
                ),
            }
        )
    keys = (
        "top1_iou",
        "neighbour_mean_iou",
        "neighbour_median_iou",
        "neighbour_max_iou",
        "neighbour_at_least_0_95",
        "random_neighbour_mean_iou",
        "condition_entropy",
    )
    report = {
        "claim_role": "diagnostic_only_conditional_ceiling",
        "split": args.split,
        "image_size": args.image_size,
        "train_cases": len(train),
        "query_cases": len(query),
        "neighbors": neighbor_count,
        "best_of_n_used": False,
        "target_used_for_selection": False,
        "summary": {
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
        },
        "strict_0_95_identifiable_fraction": float(
            np.mean([row["neighbour_max_iou"] >= 0.95 for row in rows])
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
