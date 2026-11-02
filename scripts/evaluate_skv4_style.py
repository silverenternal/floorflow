#!/usr/bin/env python3
"""Evaluate user-selected style conditioning on the skv4 test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.skv4_dataset import Skv4PlanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.distribution.sampling import sample_distribution_floorplans
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.sample_diversity import (
    bootstrap_mean,
    coverage_density,
    mmd_rbf,
    pairwise_distances,
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
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--style-bank", type=Path, default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--source-noise-scale", type=float, default=0.01)
    parser.add_argument("--style-gate", type=float, default=None)
    parser.add_argument(
        "--solver",
        choices=("euler", "shifted_euler", "heun"),
        default="euler",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    dataset = Skv4PlanDataset(
        args.skv4_root,
        args.split,
        taxonomy,
        image_size=256,
        limit=args.limit,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    style_ids = (
        torch.tensor([args.style_id], device=device, dtype=torch.long)
        if args.style_id is not None
        else None
    )
    style_gate = args.style_gate
    if style_gate is None:
        style_gate = 1.0
        if args.style_bank is not None and args.style_id is not None:
            bank = json.loads(args.style_bank.read_text(encoding="utf-8"))
            style_gate = float(bank["style_gates"][args.style_id])
    quality = []
    generated_features = []
    target_features = []
    case_rows = []
    for index in range(len(dataset)):
        target, footprint, _floor_id, name = dataset[index]
        target = target[None].to(device)
        footprint = footprint[None, None].to(device)
        target_support = torch.isin(target, room_ids)
        if not bool(target_support.any()):
            continue
        case_quality = []
        case_features = []
        for sample_index in range(args.samples):
            torch.manual_seed(args.seed + index * 1009 + sample_index)
            state, _initial = sample_distribution_floorplans(
                model,
                footprint,
                floor_ids=torch.zeros(1, dtype=torch.long, device=device),
                style_ids=style_ids,
                steps=args.flow_steps,
                noise_scale=args.source_noise_scale,
                style_gate=style_gate,
                solver=args.solver,
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
            ) * footprint[:, 0].long()
            predicted_support = torch.isin(labels[0], room_ids)
            suite = room_metric_suite(
                labels[0].cpu().numpy(),
                target[0].cpu().numpy(),
                footprint[0, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_support.cpu().numpy(),
                predicted_instance_support=predicted_support.cpu().numpy(),
                target_room_support=target_support[0].cpu().numpy(),
            )
            case_quality.append(suite["visible_semantic"]["room_macro_iou"])
            feature = floorplan_feature_vector(labels[0].cpu().numpy(), taxonomy)
            case_features.append(feature)
            generated_features.append(feature)
            quality.append(case_quality[-1])
        target_features.append(
            floorplan_feature_vector(target[0].cpu().numpy(), taxonomy)
        )
        feature_array = np.stack(case_features)
        case_rows.append(
            {
                "id": name,
                "room_iou_mean": float(np.mean(case_quality)),
                "feature_diversity": float(
                    pairwise_distances(feature_array, feature_array).mean()
                ),
            }
        )
        print(f"evaluated {index + 1}/{len(dataset)}", flush=True)
    generated = np.stack(generated_features)
    target = np.stack(target_features)
    center, scale = fit_feature_standardizer(target)
    generated_standard = transform_features(generated, center, scale)
    target_standard = transform_features(target, center, scale)
    quality_array = np.asarray(quality, dtype=np.float64)
    report = {
        "dataset": "skv4",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "condition_mode": (
            "contour_plus_style"
            if args.style_id is not None
            else "contour_only"
        ),
        "style_id": args.style_id,
        "style_bank": str(args.style_bank) if args.style_bank else None,
        "style_provenance": (
            "user_selected" if args.style_id is not None else "none"
        ),
        "best_of_n_used": False,
        "cases": len(case_rows),
        "samples_per_condition": args.samples,
        "flow_steps": args.flow_steps,
        "source_noise_scale": args.source_noise_scale,
        "style_gate": style_gate,
        "solver": args.solver,
        "sample_quality": {
            **bootstrap_mean(quality_array, seed=args.seed),
            "median": float(np.median(quality_array)),
        },
        "distribution": {
            "mmd_rbf": mmd_rbf(generated_standard, target_standard),
            **coverage_density(generated_standard, target_standard),
            "mean_within_condition_feature_distance": float(
                np.mean([row["feature_diversity"] for row in case_rows])
            ),
        },
        "rows": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["sample_quality"], indent=2))
    print(json.dumps(report["distribution"], indent=2))


if __name__ == "__main__":
    main()
