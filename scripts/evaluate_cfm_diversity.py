#!/usr/bin/env python3
"""Evaluate conditional CFM sample quality and distributional diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.sample_diversity import (
    coverage_density,
    mean_pairwise_distance,
    mmd_rbf,
)
from floorflow.flow.guided_control import (
    guided_source_from_labels,
    sample_guided_floorplans,
)
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.inference.contour_controls import predict_contour_controls
from floorflow.representation.core import (
    fit_feature_standardizer,
    floorplan_feature_vector,
    transform_features,
)
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guided-control", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=None,
    )
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    generated_features = []
    target_features = []
    quality_rows = []
    condition_rows = []
    for index in range(len(dataset)):
        target, footprint, floor_id, name = dataset[index]
        target = target[None].to(device)
        footprint = footprint[None].to(device)
        target_support = torch.isin(target, room_ids).cpu().numpy()[0]
        if not target_support.any():
            continue
        target_features.append(
            floorplan_feature_vector(target[0].cpu().numpy(), taxonomy)
        )
        case_ious = []
        case_features = []
        for sample_index in range(args.samples):
            torch.manual_seed(args.seed + index * 1009 + sample_index)
            if args.guided_control:
                source = guided_source_from_labels(
                    target,
                    footprint,
                    taxonomy,
                )
                sampled = sample_guided_floorplans(
                    model,
                    footprint,
                    source,
                    steps=args.flow_steps,
                    floor_ids=torch.tensor([floor_id], device=device),
                    noise_scale=args.noise_scale,
                )
                _, auxiliary = model.forward_with_aux(
                    sampled,
                    torch.ones(1, device=device),
                    footprint,
                    floor_ids=torch.tensor([floor_id], device=device),
                )
                labels = room_ids[
                    auxiliary["room_partition_logits"].argmax(dim=1)
                ] * footprint.long()
            else:
                controls = predict_contour_controls(
                    model,
                    footprint,
                    taxonomy,
                    floor_ids=torch.tensor([floor_id], device=device),
                    flow_steps=args.flow_steps,
                    structure_size=64,
                    furniture_size=96,
                    support_size=96,
                )
                labels = gate_room_labels_to_program(
                    room_ids[controls.room_logits.argmax(dim=1)],
                    controls.program,
                    taxonomy.room_ids,
                ) * footprint.long()
            labels_np = labels[0].cpu().numpy()
            suite = room_metric_suite(
                labels_np,
                target[0].cpu().numpy(),
                footprint[0].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(labels[0], room_ids).cpu().numpy(),
                predicted_instance_support=torch.isin(labels[0], room_ids).cpu().numpy(),
                target_room_support=target_support,
            )
            case_ious.append(suite["visible_semantic"]["room_macro_iou"])
            feature = floorplan_feature_vector(labels_np, taxonomy)
            case_features.append(feature)
            generated_features.append(feature)
            quality_rows.append(
                {
                    "id": name,
                    "sample": sample_index,
                    "room_macro_iou": case_ious[-1],
                    **suite["matched_instance"],
                }
            )
        feature_array = np.stack(case_features)
        condition_rows.append(
            {
                "id": name,
                "iou_mean": float(np.mean(case_ious)),
                "feature_diversity": mean_pairwise_distance(feature_array),
            }
        )
        print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
    generated = np.stack(generated_features)
    target = np.stack(target_features)
    center, scale = fit_feature_standardizer(target)
    generated_standard = transform_features(generated, center, scale)
    target_standard = transform_features(target, center, scale)
    quality = np.asarray(
        [row["room_macro_iou"] for row in quality_rows],
        dtype=np.float64,
    )
    distribution = coverage_density(generated_standard, target_standard)
    report = {
        "dataset": "CubiCasa",
        "split": args.split,
        "conditions": len(target_features),
        "samples_per_condition": args.samples,
        "flow_steps": args.flow_steps,
        "guided_control": args.guided_control,
        "best_of_n_used": False,
        "sample_quality": {
            "mean": float(quality.mean()),
            "median": float(np.median(quality)),
            "standard_error": float(
                quality.std(ddof=1) / np.sqrt(len(quality))
            ),
        },
        "distribution": {
            "mmd_rbf": mmd_rbf(generated_standard, target_standard),
            **distribution,
            "mean_within_condition_feature_distance": float(
                np.mean([row["feature_diversity"] for row in condition_rows])
            ),
        },
        "cases": condition_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["sample_quality"], indent=2))
    print(json.dumps(report["distribution"], indent=2))


if __name__ == "__main__":
    main()
