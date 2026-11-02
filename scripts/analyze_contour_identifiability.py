#!/usr/bin/env python3
"""Quantify how much room layout is identifiable from contour alone."""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.identifiability import (
    ContourRecord,
    contour_descriptor,
    duplicate_contour_layout_ious,
    exact_group_entropy,
    macro_iou,
    nearest_indices,
    program_neighbor_entropy,
    room_labels_at_size,
    room_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--train-limit", type=int, default=5126)
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--test-limit", type=int, default=191)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collect(
    root: Path,
    cache: Path,
    split: str,
    limit: int,
    taxonomy,
) -> list[ContourRecord]:
    dataset = CubiCasaFloorplanDataset(
        root,
        split,
        image_size=256,
        cache_root=cache,
        limit=limit,
        augment=False,
    )
    records = []
    for index in range(len(dataset)):
        target, footprint, _floor_id, name = dataset[index]
        records.append(
            ContourRecord(
                name=name,
                exact_hash=zlib.crc32(footprint.numpy().tobytes()),
                descriptor=contour_descriptor(footprint),
                room_labels=room_labels_at_size(target, 64),
                room_program=room_program(target, taxonomy).flatten(),
            )
        )
    return records


def summarize_iou(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "fraction_ge_095": float((array >= 0.95).mean()),
    }


def main() -> None:
    args = parse_args()
    first = CubiCasaFloorplanDataset(
        args.data_root,
        args.test_split,
        image_size=256,
        cache_root=args.cache_root,
        limit=1,
        augment=False,
    )
    taxonomy = first.taxonomy
    train = collect(
        args.data_root,
        args.cache_root,
        "train",
        args.train_limit,
        taxonomy,
    )
    test = collect(
        args.data_root,
        args.cache_root,
        args.test_split,
        args.test_limit,
        taxonomy,
    )
    device = torch.device(args.device)
    train_descriptors = torch.stack(
        [record.descriptor for record in train]
    ).to(device)
    test_descriptors = torch.stack(
        [record.descriptor for record in test]
    ).to(device)
    neighbors, distances = nearest_indices(
        test_descriptors,
        train_descriptors,
        k=args.neighbors,
    )
    top1_iou = [
        macro_iou(
            train[int(neighbors[index, 0])].room_labels,
            test[index].room_labels,
            taxonomy,
        )
        for index in range(len(test))
    ]
    histogram_counts, histogram_edges = np.histogram(
        top1_iou,
        bins=20,
        range=(0.0, 1.0),
    )
    train_programs = torch.stack([record.room_program for record in train])
    test_programs = torch.stack([record.room_program for record in test])
    program_entropy = program_neighbor_entropy(
        test_programs,
        train_programs,
        neighbors,
    )
    train_hashes = Counter(record.exact_hash for record in train)
    exact_matches = sum(
        int(train_hashes.get(record.exact_hash, 0) > 0)
        for record in test
    )
    report = {
        "scope": "data-only diagnostic; neighbours selected by contour only",
        "train_cases": len(train),
        "test_cases": len(test),
        "exact_contour": {
            "train_duplicate_groups": sum(
                1 for count in train_hashes.values() if count > 1
            ),
            "test_with_train_exact_match_fraction": exact_matches / max(1, len(test)),
            "train_program_entropy": exact_group_entropy(train),
            "test_program_entropy": exact_group_entropy(test),
            "train_layout_iou": duplicate_contour_layout_ious(
                train,
                taxonomy,
            ),
            "test_layout_iou": duplicate_contour_layout_ious(
                test,
                taxonomy,
            ),
        },
        "nearest_train_layout": {
            "descriptor_distance_mean": float(distances[:, 0].mean()),
            "room_macro_iou": summarize_iou(top1_iou),
            "top1_room_macro_iou_histogram": {
                "counts": histogram_counts.astype(int).tolist(),
                "edges": histogram_edges.tolist(),
            },
            "per_case_top1_room_macro_iou": [
                float(value) for value in top1_iou
            ],
            "knn_program_binary_entropy_mean": float(program_entropy.mean()),
            "knn_program_binary_entropy_p90": float(
                np.quantile(program_entropy, 0.90)
            ),
        },
        "interpretation": {
            "contour_only": "Nearest-train IoU is a retrieval proxy, not a model upper bound.",
            "required_next": "Compare this frontier with learned contour-only and sparse models.",
        },
        "device_requested": args.device,
        "best_of_n_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
