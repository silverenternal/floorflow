#!/usr/bin/env python3
"""Evaluate raster-conditioned room controls with the DMD overlay decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from train_raster_room_control import raster_batch

from floorflow.data.dataset import (
    CubiCasaFloorplanDataset,
    CubiCasaRasterControlDataset,
)
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.room_segmenter_prior import load_room_segmenter_prior
from floorflow.representation.core import extract_room_instances

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raster-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_1024_v1/best.pt",
    )
    parser.add_argument(
        "--input-rendering",
        choices=("native", "line", "nonroom"),
        default="native",
    )
    parser.add_argument(
        "--contour-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/room_segmenter_r18_uniform_full40_v2/best.pt",
    )
    parser.add_argument(
        "--distribution-checkpoint",
        type=Path,
        default=ROOT
        / "outputs/train/distribution_dmd_adversarial_datamap4_v11/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--room-prior-weight", type=float, default=2.0)
    parser.add_argument(
        "--blend-weights",
        nargs="+",
        type=float,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/raster_room_control_val491_v1.json",
    )
    return parser.parse_args()


def count_instances(labels: torch.Tensor, taxonomy) -> int:
    array = labels.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return len(extract_room_instances(array, taxonomy, min_area=8))


def metric_row(
    labels: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy,
) -> dict[str, float | int]:
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    support = torch.isin(labels, room_ids)
    target_support = torch.isin(target, room_ids)
    suite = room_metric_suite(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=support.cpu().numpy(),
        predicted_instance_support=support.cpu().numpy(),
        target_room_support=target_support.cpu().numpy(),
    )
    instance = suite["matched_instance"]
    return {
        "room_macro_iou": float(
            suite["visible_semantic"]["room_macro_iou"]
        ),
        "instance_geometric_iou": float(
            instance["matched_geometric_iou"]
        ),
        "instance_room_f1": float(instance["instance_f1_at_threshold"]),
        "instance_type_accuracy": float(instance["matched_type_accuracy"]),
        "class_constrained_iou": float(
            instance["matched_class_constrained_iou"]
        ),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
        "predicted_instances": count_instances(labels, taxonomy),
        "target_instances": count_instances(target, taxonomy),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    raster_model = load_raster_room_model(
        args.raster_checkpoint,
        taxonomy := CubiCasaFloorplanDataset(
            args.data_root,
            args.split,
            cache_root=args.cache_root,
            limit=1,
        ).taxonomy,
        device,
    )
    contour_model, _ = load_room_segmenter_prior(
        args.contour_checkpoint,
        taxonomy,
        device,
    )
    distribution, _, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    dataset_class = (
        CubiCasaRasterControlDataset
        if args.input_rendering in {"native", "line"}
        else CubiCasaFloorplanDataset
    )
    dataset_kwargs = (
        {"image_mode": args.input_rendering}
        if dataset_class is CubiCasaRasterControlDataset
        else {}
    )
    dataset = dataset_class(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        **dataset_kwargs,
    )
    stop = min(len(dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    style_id = torch.zeros(1, dtype=torch.long, device=device)
    weights = tuple(dict.fromkeys(float(value) for value in args.blend_weights))
    rows = {str(weight): [] for weight in weights}
    rows["raster_raw"] = []
    for batch in loader:
        if len(batch) == 5:
            target, footprint, control, floor_ids, names = batch
            control = control.to(device)
        else:
            target, footprint, floor_ids, names = batch
            target = target.to(device)
            footprint = footprint.to(device)
            control = raster_batch(target, footprint, taxonomy)
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        style_ids = style_id.expand(footprint.shape[0])
        raster_logits = raster_model(footprint, control_image=control)
        contour_logits = contour_model(footprint)
        raster_probability = raster_logits.log_softmax(dim=1)
        contour_probability = contour_logits.log_softmax(dim=1)
        room_tensor = torch.tensor(taxonomy.room_ids, device=device)
        raw_labels = room_tensor[raster_logits.argmax(dim=1)] * footprint.long()
        for index, name in enumerate(names):
            rows["raster_raw"].append(
                {
                    "case": name,
                    **metric_row(
                        raw_labels[index],
                        target[index],
                        footprint[index],
                        taxonomy,
                    ),
                }
            )
            for weight in weights:
                prior = (
                    (1.0 - weight) * contour_probability[index : index + 1]
                    + weight * raster_probability[index : index + 1]
                )
                decoded = decode_overlay(
                    distribution,
                    None,
                    None,
                    footprint[index : index + 1, None],
                    taxonomy,
                    floor_ids=floor_ids[index : index + 1],
                    style_ids=style_ids[index : index + 1],
                    room_prior_logits=prior,
                    room_prior_weight=args.room_prior_weight,
                    room_prior_confidence_threshold=1.0,
                    settings=settings,
                )
                rows[str(weight)].append(
                    {
                        "case": name,
                        **metric_row(
                            decoded["labels"][0],
                            target[index],
                            footprint[index],
                            taxonomy,
                        ),
                    }
                )
    metrics = {
        name: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "case"
        }
        for name, values in rows.items()
    }
    baseline = rows[str(weights[0])]
    paired = {
        name: {
            key: paired_delta(
                values,
                baseline,
                key,
                iterations=args.bootstrap_iterations,
                seed=3101,
            )
            for key in (
                "room_macro_iou",
                "instance_geometric_iou",
                "instance_room_f1",
                "instance_type_accuracy",
            )
        }
        for name, values in rows.items()
        if name != str(weights[0])
    }
    report = {
        "schema_version": "raster-room-control-eval-v1",
        "scope": "CubiCasa raster-conditioned reconstruction; no room labels in input",
        "input_rendering": args.input_rendering,
        "split": args.split,
        "cases": stop - args.offset,
        "raster_checkpoint": str(args.raster_checkpoint),
        "contour_checkpoint": str(args.contour_checkpoint),
        "distribution_checkpoint": str(args.distribution_checkpoint),
        "blend_definition": (
            "(1-w) * contour_log_probability + "
            "w * raster_log_probability before DMD overlay"
        ),
        "metrics": metrics,
        "paired_vs_contour_overlay": paired,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
