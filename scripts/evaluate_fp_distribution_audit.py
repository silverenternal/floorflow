#!/usr/bin/env python3
"""Batched distribution audit for the frozen contour-only sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.distribution.sampling import sample_distribution_floorplans
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.sample_diversity import (
    bootstrap_mean,
    coverage_density,
    mean_pairwise_distance,
    mmd_rbf,
)
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.representation.core import (
    fit_feature_standardizer,
    floorplan_feature_vector,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=501)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--source-noise-scale", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
        write_cache=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    generated_features = []
    target_features = []
    quality = []
    coverage_values = []
    case_ids = []
    for batch_index, (target, footprint, floor_ids, names) in enumerate(loader):
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        target_support = torch.isin(target, room_ids)
        batch_generated = [[] for _ in range(target.shape[0])]
        for sample_index in range(args.samples):
            torch.manual_seed(
                args.seed + batch_index * 1009 + sample_index
            )
            state, _ = sample_distribution_floorplans(
                model,
                footprint,
                steps=args.flow_steps,
                floor_ids=floor_ids,
                noise_scale=args.source_noise_scale,
                solver="euler",
                source_mode="blend",
            )
            scores = state[:, room_ids - 1]
            labels = room_ids[scores.argmax(dim=1)]
            program = scores.amax(dim=(2, 3)) > 0.1
            missing = ~program.any(dim=1)
            if missing.any():
                fallback = scores.amax(dim=(2, 3)).argmax(dim=1)
                missing_index = torch.nonzero(
                    missing,
                    as_tuple=False,
                ).flatten()
                program[missing_index, fallback[missing_index]] = True
            labels = gate_room_labels_to_program(
                labels,
                program,
                taxonomy.room_ids,
            ) * footprint.long()
            for index, name in enumerate(names):
                predicted_support = torch.isin(labels[index], room_ids)
                suite = room_metric_suite(
                    labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=predicted_support.cpu().numpy(),
                    predicted_instance_support=predicted_support.cpu().numpy(),
                    target_room_support=target_support[index].cpu().numpy(),
                )
                quality.append(
                    suite["visible_semantic"]["room_macro_iou"]
                )
                coverage_values.append(labels[index].ne(0).float().mean().item())
                feature = floorplan_feature_vector(
                    labels[index].cpu().numpy(),
                    taxonomy,
                )
                batch_generated[index].append(feature)
                if sample_index == 0:
                    case_ids.append(str(name))
                    target_features.append(
                        floorplan_feature_vector(
                            target[index].cpu().numpy(),
                            taxonomy,
                        )
                    )
        generated_features.extend(np.stack(values) for values in batch_generated)
        print(
            f"evaluated {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
            flush=True,
        )
    generated = np.stack(generated_features)
    target = np.stack(target_features)
    center, scale = fit_feature_standardizer(target)
    generated_standard = transform_features(generated, center, scale)
    target_standard = transform_features(target, center, scale)
    quality_array = np.asarray(quality, dtype=np.float64)
    diversity = np.asarray(
        [mean_pairwise_distance(values) for values in generated_features]
    )
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "conditions": len(target),
        "samples_per_condition": args.samples,
        "quality": bootstrap_mean(quality_array, seed=args.seed),
        "room_coverage_mean": float(np.mean(coverage_values)),
        "distribution": {
            "mmd_rbf": mmd_rbf(generated_standard, target_standard),
            **coverage_density(generated_standard, target_standard),
            "within_condition_diversity": float(diversity.mean()),
        },
        "best_of_n_used": False,
        "target_derived_control": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.feature_output,
        generated=generated_standard,
        target=target_standard,
        case_ids=np.asarray(case_ids),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
