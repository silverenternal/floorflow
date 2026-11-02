#!/usr/bin/env python3
"""Evaluate fresh contours through the distribution-CFM sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--source-noise-scale", type=float, default=0.0)
    parser.add_argument(
        "--sample-source",
        choices=("blend", "pixel", "element"),
        default="blend",
    )
    parser.add_argument("--style-gate", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        choices=("euler", "shifted_euler", "heun"),
        default="euler",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--style-ids", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2 to measure conditional diversity")
    return args


def evaluate_style(model, dataset, taxonomy, args, device, style_id):
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    style_ids = None
    if style_id is not None:
        style_ids = torch.tensor([style_id], device=device, dtype=torch.long)
    generated_features = []
    target_features = []
    quality = []
    case_rows = []
    for index in range(len(dataset)):
        target, footprint, floor_id, name = dataset[index]
        target = target[None].to(device)
        footprint = footprint[None].to(device)
        target_support = torch.isin(target, room_ids)
        if not bool(target_support.any()):
            continue
        target_features.append(
            floorplan_feature_vector(target[0].cpu().numpy(), taxonomy)
        )
        case_iou = []
        case_features = []
        for sample_index in range(args.samples):
            torch.manual_seed(args.seed + index * 1009 + sample_index)
            state, _ = sample_distribution_floorplans(
                model,
                footprint,
                steps=args.flow_steps,
                floor_ids=torch.tensor([floor_id], device=device),
                style_ids=style_ids,
                noise_scale=args.source_noise_scale,
                style_gate=args.style_gate,
                solver=args.solver,
                source_mode=args.sample_source,
            )
            scores = state[:, room_ids - 1]
            labels = room_ids[scores.argmax(dim=1)]
            program = scores.amax(dim=(2, 3)) > 0.1
            missing = ~program.any(dim=1)
            if missing.any():
                fallback = scores.amax(dim=(2, 3)).argmax(dim=1)
                program = program.clone()
                program[missing, fallback[missing]] = True
            labels = gate_room_labels_to_program(
                labels,
                program,
                taxonomy.room_ids,
            ) * footprint.long()
            predicted_support = torch.isin(labels[0], room_ids)
            suite = room_metric_suite(
                labels[0].cpu().numpy(),
                target[0].cpu().numpy(),
                footprint[0].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_support.cpu().numpy(),
                predicted_instance_support=predicted_support.cpu().numpy(),
                target_room_support=target_support[0].cpu().numpy(),
            )
            case_iou.append(suite["visible_semantic"]["room_macro_iou"])
            feature = floorplan_feature_vector(labels[0].cpu().numpy(), taxonomy)
            case_features.append(feature)
            generated_features.append(feature)
            quality.append(case_iou[-1])
        feature_array = np.stack(case_features)
        case_rows.append(
            {
                "id": name,
                "iou_mean": float(np.mean(case_iou)),
                "feature_diversity": mean_pairwise_distance(feature_array),
            }
        )
        print(
            f"style={style_id} evaluated {index + 1}/{len(dataset)}",
            flush=True,
        )
    generated = np.stack(generated_features)
    target = np.stack(target_features)
    center, scale = fit_feature_standardizer(target)
    generated_standard = transform_features(generated, center, scale)
    target_standard = transform_features(target, center, scale)
    quality_array = np.asarray(quality, dtype=np.float64)
    within_condition_diversity = float(
        np.mean([row["feature_diversity"] for row in case_rows])
    )
    diversity_bootstrap = bootstrap_mean(
        np.asarray([row["feature_diversity"] for row in case_rows]),
        seed=args.seed + 104729,
    )
    target_diversity = mean_pairwise_distance(target_standard)
    report = {
        "style_id": style_id,
        "style_provenance": "user_selected" if style_id is not None else "none",
        "condition_mode": (
            "contour_plus_style" if style_id is not None else "contour_only"
        ),
        "conditions": len(target_features),
        "samples_per_condition": args.samples,
        "flow_steps": args.flow_steps,
        "source_noise_scale": args.source_noise_scale,
        "style_gate": args.style_gate,
        "solver": args.solver,
        "sample_source": args.sample_source,
        "sample_quality": {
            **bootstrap_mean(quality_array, seed=args.seed),
            "median": float(np.median(quality_array)),
            "standard_error": float(
                quality_array.std(ddof=1) / np.sqrt(len(quality_array))
            ),
        },
        "distribution": {
            "mmd_rbf": mmd_rbf(generated_standard, target_standard),
            **coverage_density(generated_standard, target_standard),
            "mean_within_condition_feature_distance": within_condition_diversity,
            "within_condition_feature_distance_bootstrap": diversity_bootstrap,
            "unconditional_target_cross_plan_feature_distance": target_diversity,
            "conditional_to_target_diversity_ratio": (
                within_condition_diversity / target_diversity
                if target_diversity > 0
                else 0.0
            ),
        },
        "cases": case_rows,
    }
    if args.feature_output is not None:
        report["_feature_export"] = {
            "generated": generated_standard,
            "target": target_standard,
            "case_ids": np.asarray([row["id"] for row in case_rows]),
        }
    return report


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    training_args = payload.get("training_args", {})
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(training_args.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    style_ids = args.style_ids
    if style_ids is None:
        style_ids = [args.style_id]
    style_reports = {
        str(style_id): evaluate_style(
            model,
            dataset,
            taxonomy,
            args,
            device,
            style_id,
        )
        for style_id in style_ids
    }
    if args.feature_output is not None:
        for style_id, report in style_reports.items():
            exported = report.pop("_feature_export", None)
            if exported is None:
                continue
            output = args.feature_output
            if len(style_reports) > 1:
                output = output.with_name(
                    f"{output.stem}_style{style_id}{output.suffix}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **exported)
    quality_means = [
        report["sample_quality"]["mean"]
        for report in style_reports.values()
    ]
    coverages = [
        report["distribution"]["coverage"]
        for report in style_reports.values()
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": "CubiCasa",
        "split": args.split,
        "best_of_n_used": False,
        "style_selection": "user supplied; no per-case target style",
        "style_ids": style_ids,
        "sample_source": args.sample_source,
        "styles": style_reports,
        "aggregate": {
            "quality_mean": float(np.mean(quality_means)),
            "quality_worst": float(np.min(quality_means)),
            "quality_best": float(np.max(quality_means)),
            "coverage_mean": float(np.mean(coverages)),
            "coverage_worst": float(np.min(coverages)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
