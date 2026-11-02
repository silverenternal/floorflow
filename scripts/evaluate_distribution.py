#!/usr/bin/env python3
"""Evaluate conditional layout distributions instead of cherry-picking best samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.representation.core import (
    contour_descriptor,
    coverage_density,
    energy_distance,
    fit_feature_standardizer,
    floorplan_feature_vector,
    mmd_rbf,
    transform_features,
)
from floorflow.inference.sampling_policy import SamplingPolicy, expected_sample_statistics
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.data.taxonomy import FloorplanTaxonomy


def _bootstrap_ci(
    values: list[float],
    iterations: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [data[rng.integers(0, len(data), len(data))].mean() for _ in range(iterations)]
    )
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v2")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--reference-limit", type=int, default=1000)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument(
        "--room-source",
        choices=("mask", "partition", "semantic", "hierarchical", "polygon"),
        default="partition",
    )
    parser.add_argument("--object-source", choices=("auto", "semantic", "heads"), default="heads")
    parser.add_argument("--opening-threshold", type=float, default=0.7)
    parser.add_argument("--furniture-threshold", type=float, default=0.7)
    parser.add_argument("--structural-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/distribution/report.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    policy = SamplingPolicy(
        purpose="distribution",
        candidates_per_condition=args.samples,
        seed=args.seed,
    )
    model, config, taxonomy, _state = load_checkpoint_model(args.checkpoint, args.device)
    train = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=128,
        cache_root=args.cache_root,
        limit=args.reference_limit,
        write_cache=False,
    )
    test = CubiCasaFloorplanDataset(
        args.data_root,
        "test",
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        write_cache=False,
    )
    train_features = np.stack([contour_descriptor(train[index][1].numpy()) for index in range(len(train))])
    train_targets = [
        floorplan_feature_vector(train[index][0].numpy(), taxonomy) for index in range(len(train))
    ]
    feature_center, feature_scale = fit_feature_standardizer(np.stack(train_targets))
    normalized_train = train_features / np.linalg.norm(train_features, axis=1, keepdims=True).clip(min=1.0e-6)
    references = []
    case_names = []
    for index in range(len(test)):
        _target, footprint, _floor_id, name = test[index]
        descriptor = contour_descriptor(footprint.numpy())
        scores = normalized_train @ (
            descriptor / max(np.linalg.norm(descriptor), 1.0e-6)
        )
        nearest = np.argpartition(
            -scores,
            kth=min(args.neighbors, len(scores) - 1),
        )[: args.neighbors]
        references.append(
            transform_features(
                np.stack([train_targets[item] for item in nearest]),
                feature_center,
                feature_scale,
            )
        )
        case_names.append(name)
    items = [test[index] for index in range(len(test))]
    targets = torch.stack([item[0] for item in items]).to(args.device)
    footprints = torch.stack([item[1] for item in items]).to(args.device)
    floor_ids = torch.tensor(
        [int(item[2]) for item in items],
        device=args.device,
    )
    generated_by_case = [[] for _ in items]
    sample_rows_by_case = [[] for _ in items]
    for _ in range(args.samples):
        sampled, _ = sample_floorplans(
            model,
            footprints,
            steps=args.steps,
            floor_ids=floor_ids,
            objective_mode=config.get("objective_mode", "velocity"),
            semantic_decode_weight=0.0,
            guidance_scale=args.guidance_scale,
        )
        if args.structured_output:
            decoded, _room_support = decode_structured_floorplan(
                model,
                targets,
                footprints,
                taxonomy,
                floor_ids=floor_ids,
                opening_threshold=args.opening_threshold,
                furniture_threshold=args.furniture_threshold,
                structural_gate=args.structural_gate,
                object_source=args.object_source,
                room_source=args.room_source,
                return_room_support=True,
                state=sampled,
            )
        else:
            decoded = compose_known_exterior(
                decode_floorplans(sampled),
                footprints,
                opening_ids=taxonomy.opening_ids,
            )
        for index in range(len(items)):
            labels = decoded[index].cpu().numpy()
            generated_by_case[index].append(
                floorplan_feature_vector(labels, taxonomy)
            )
            sample_rows_by_case[index].append(
                floorplan_metrics(
                    decoded[index : index + 1],
                    targets[index : index + 1],
                    footprints[index : index + 1],
                    taxonomy,
                )
            )
    rows = []
    for index in range(len(items)):
        reference = references[index]
        generated_features = generated_by_case[index]
        sample_rows = sample_rows_by_case[index]
        generated = transform_features(np.stack(generated_features), feature_center, feature_scale)
        coverage, density = coverage_density(reference, generated)
        sample_room_iou = np.asarray([row["room_macro_iou"] for row in sample_rows])
        statistics = expected_sample_statistics(sample_room_iou)
        rows.append(
            {
                "case": case_names[index],
                "mmd": mmd_rbf(reference, generated),
                "energy": energy_distance(reference, generated),
                "coverage": coverage,
                "density": density,
                "diversity": float(
                    np.sqrt(((generated[:, None, :] - generated[None, :, :]) ** 2).sum(axis=-1))[
                        np.triu_indices(len(generated), k=1)
                    ].mean()
                ),
                "expected_room_macro_iou": statistics["mean"],
                "median_room_macro_iou": statistics["median"],
                "room_macro_iou_standard_error": statistics["standard_error"],
            }
        )
    report = {
        "claim_role": "conditional_distribution",
        "best_of_n_used": False,
        "sampling_policy": policy.manifest(),
        "test_samples": len(rows),
        "generated_per_condition": args.samples,
        "reference_neighbors_per_condition": args.neighbors,
        "feature_scaling": "train_median_iqr",
        "mmd_bandwidth": "pooled_median_heuristic",
        "guidance_scale": args.guidance_scale,
        "structured_output": args.structured_output,
        "room_source": args.room_source,
        "object_source": args.object_source,
        "mmd": float(np.mean([row["mmd"] for row in rows])),
        "energy": float(np.mean([row["energy"] for row in rows])),
        "coverage": float(np.mean([row["coverage"] for row in rows])),
        "density": float(np.mean([row["density"] for row in rows])),
        "diversity": float(np.mean([row["diversity"] for row in rows])),
        "expected_room_macro_iou": float(np.mean([row["expected_room_macro_iou"] for row in rows])),
        "median_room_macro_iou": float(np.mean([row["median_room_macro_iou"] for row in rows])),
        "rows": rows,
    }
    report["confidence_intervals"] = {
        key: _bootstrap_ci([row[key] for row in rows], seed=args.seed)
        for key in ("mmd", "energy", "coverage", "density", "diversity", "expected_room_macro_iou")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mmd",
                    "energy",
                    "coverage",
                    "density",
                    "diversity",
                    "expected_room_macro_iou",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
