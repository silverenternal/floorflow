#!/usr/bin/env python3
"""Build a train-only room-design style bank from floorplan descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.data.skv4_dataset import Skv4PlanDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.representation.core import (
    fit_feature_standardizer,
    floorplan_feature_vector,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--dataset", choices=("cubicasa", "skv4"), default="cubicasa")
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument("--taxonomy", type=Path, default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--num-styles", type=int, default=8)
    parser.add_argument("--method", choices=("kmeans", "pc-quantile"), default="kmeans")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "cubicasa":
        dataset = CubiCasaFloorplanDataset(
            args.data_root,
            "train",
            image_size=256,
            cache_root=args.cache_root,
            augment=False,
        )
    else:
        if args.taxonomy is None:
            raise SystemExit("--taxonomy is required for skv4")
        dataset = Skv4PlanDataset(
            args.skv4_root,
            "train",
            FloorplanTaxonomy.read(args.taxonomy),
            image_size=256,
        )
    features = []
    names = []
    for index in range(len(dataset)):
        target, _footprint, _floor_id, name = dataset[index]
        features.append(
            floorplan_feature_vector(target.numpy(), dataset.taxonomy)
        )
        names.append(name)
    feature_array = np.nan_to_num(np.stack(features).astype(np.float64))
    center, scale = fit_feature_standardizer(feature_array)
    standardized = transform_features(feature_array, center, scale)
    if args.method == "kmeans":
        model = KMeans(
            n_clusters=args.num_styles,
            n_init=10,
            random_state=args.seed,
        )
        assignment = model.fit_predict(standardized)
        order = sorted(
            range(args.num_styles),
            key=lambda index: tuple(model.cluster_centers_[index].tolist()),
        )
        remap = {old: new for new, old in enumerate(order)}
        centroids = model.cluster_centers_[order]
    else:
        component = PCA(n_components=1, random_state=args.seed).fit_transform(
            standardized
        )[:, 0]
        order = np.argsort(component)
        assignment = np.empty(len(order), dtype=np.int64)
        for rank, index in enumerate(order):
            assignment[index] = min(
                args.num_styles - 1,
                rank * args.num_styles // max(1, len(order)),
            )
        remap = {index: index for index in range(args.num_styles)}
        centroids = np.stack(
            [
                standardized[assignment == index].mean(axis=0)
                for index in range(args.num_styles)
            ]
        )
    assignments = {
        name: remap[int(style)]
        for name, style in zip(names, assignment)
    }
    report = {
        "num_styles": int(args.num_styles),
        "source": f"train-only {args.dataset} floorplan structural descriptors",
        "seed": args.seed,
        "method": args.method,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "centroids_standardized": centroids.tolist(),
        "cluster_sizes": [
            int(np.sum(assignment == index))
            for index in range(args.num_styles)
        ],
        "assignments": {"train": assignments},
        "inference_rule": "style id is supplied explicitly by the user or deployment system",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("num_styles", "cluster_sizes", "inference_rule")}, indent=2))


if __name__ == "__main__":
    main()
