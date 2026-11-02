#!/usr/bin/env python3
"""Train a contour-only style predictor from a train-only style bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.representation.core import (
    contour_descriptor,
    fit_feature_standardizer,
    floorplan_feature_vector,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def contour_rows(dataset) -> np.ndarray:
    return np.stack(
        [
            contour_descriptor(dataset[index][1].numpy().astype(bool))
            for index in range(len(dataset))
        ]
    )


def nearest_style_labels(dataset, bank) -> np.ndarray:
    center = np.asarray(bank["feature_center"])
    scale = np.asarray(bank["feature_scale"])
    centroids = np.asarray(bank["centroids_standardized"])
    labels = []
    for index in range(len(dataset)):
        target, _footprint, _floor, _name = dataset[index]
        feature = floorplan_feature_vector(target.numpy(), dataset.taxonomy)
        standardized = transform_features(feature[None], center, scale)
        labels.append(int(np.argmin(((centroids - standardized) ** 2).sum(axis=1))))
    return np.asarray(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()
    bank = json.loads(args.style_bank.read_text(encoding="utf-8"))
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    val = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    train_names = {name: style for name, style in bank["assignments"]["train"].items()}
    train_labels = np.asarray(
        [train_names[str(train[index][3])] for index in range(len(train))],
        dtype=np.int64,
    )
    train_contours = contour_rows(train)
    val_contours = contour_rows(val)
    center, scale = fit_feature_standardizer(train_contours)
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
    )
    classifier.fit(
        transform_features(train_contours, center, scale),
        train_labels,
    )
    val_labels = nearest_style_labels(val, bank)
    predicted = classifier.predict(transform_features(val_contours, center, scale))
    report = {
        "input_mode": "contour_only",
        "style_bank": str(args.style_bank),
        "num_styles": int(bank["num_styles"]),
        "train_cases": len(train_labels),
        "val_cases": len(val_labels),
        "val_style_accuracy": float((predicted == val_labels).mean()),
        "target_used_for_inference": False,
        "target_used_for_val_label": True,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "num_styles": report["num_styles"],
                "val_style_accuracy": report["val_style_accuracy"],
                "input_mode": report["input_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
