#!/usr/bin/env python3
"""Evaluate a train-only style-constrained contour retrieval baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.sample_diversity import coverage_density, mmd_rbf
from floorflow.representation.core import (
    contour_descriptor,
    fit_feature_standardizer,
    floorplan_feature_vector,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--style-bank",
        type=Path,
        default=Path("outputs/style_banks/cubicasa_train_v1.json"),
    )
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def descriptors_and_names(dataset) -> tuple[np.ndarray, list[str]]:
    descriptors = []
    names = []
    for index in range(len(dataset)):
        _target, footprint, _floor, name = dataset[index]
        descriptors.append(contour_descriptor(footprint.numpy().astype(bool)))
        names.append(str(name))
    return np.stack(descriptors).astype(np.float32), names


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
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    train_descriptors, train_names = descriptors_and_names(train)
    train_assignments = bank["assignments"]["train"]
    indices = np.asarray(
        [
            index
            for index, name in enumerate(train_names)
            if int(train_assignments[name]) == args.style_id
        ],
        dtype=np.int64,
    )
    if not len(indices):
        raise SystemExit(f"style {args.style_id} has no train cases")
    train_descriptors = train_descriptors[indices]
    train_descriptors /= np.linalg.norm(
        train_descriptors,
        axis=1,
        keepdims=True,
    ).clip(1e-8)

    loader = DataLoader(
        Subset(val, range(min(args.limit, len(val)))),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    room_ids = torch.tensor(val.taxonomy.room_ids)
    quality = []
    target_features = []
    generated_features = []
    for target, footprint, _floor, _name in loader:
        descriptor = contour_descriptor(footprint[0].numpy().astype(bool))
        descriptor = descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)
        nearest = int(np.argmax(train_descriptors @ descriptor))
        retrieved = train[int(indices[nearest])][0]
        predicted = retrieved * footprint[0].long()
        suite = room_metric_suite(
            predicted.numpy(),
            target[0].numpy(),
            footprint[0].numpy(),
            val.taxonomy,
            predicted_room_support=torch.isin(predicted, room_ids).numpy(),
            predicted_instance_support=torch.isin(predicted, room_ids).numpy(),
            target_room_support=torch.isin(target[0], room_ids).numpy(),
        )
        quality.append(suite["visible_semantic"]["room_macro_iou"])
        target_features.append(
            floorplan_feature_vector(target[0].numpy(), val.taxonomy)
        )
        generated_features.append(
            floorplan_feature_vector(predicted.numpy(), val.taxonomy)
        )
    target_array = np.stack(target_features)
    generated_array = np.stack(generated_features)
    center, scale = fit_feature_standardizer(target_array)
    target_standard = transform_features(target_array, center, scale)
    generated_standard = transform_features(generated_array, center, scale)
    distribution = coverage_density(target_standard, generated_standard)
    report = {
        "baseline": "train-only style-constrained contour retrieval",
        "split": args.split,
        "cases": len(quality),
        "style_id": args.style_id,
        "style_bank": str(args.style_bank),
        "best_of_n_used": False,
        "target_used_for_retrieval": False,
        "room_macro_iou": float(np.mean(quality)),
        "coverage": distribution["coverage"],
        "density": distribution["density"],
        "mmd_rbf": mmd_rbf(target_standard, generated_standard),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
