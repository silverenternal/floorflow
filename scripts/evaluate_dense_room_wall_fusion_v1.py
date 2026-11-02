#!/usr/bin/env python3
"""Evaluate deterministic dense-room and predicted-wall instance fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import (
    generation_plausibility,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import load_raster_control_bank
from floorflow.inference.dense_room_wall_fusion import fuse_dense_room_walls

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/raster_full_cad_bank_frozen_room_screen_v4/best.pt"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/dense_room_wall_fusion_cal64_v1.json",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _arms() -> list[tuple[str, float | None]]:
    arms = [("dense", None), ("boundary_wall", None)]
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        arms.append((f"predicted_wall_{threshold:.1f}", threshold))
        arms.append((f"union_{threshold:.1f}", threshold))
    return arms


def _room_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids)
    predicted_support = np.isin(prediction, room_ids)
    target_support = np.isin(target, room_ids)
    suite = room_metric_suite(
        prediction,
        target,
        footprint,
        taxonomy,
        predicted_room_support=predicted_support,
        predicted_instance_support=predicted_support,
        target_room_support=target_support,
    )
    instance = suite["matched_instance"]
    return {
        "room_macro_iou": float(
            suite["visible_semantic"]["room_macro_iou"]
        ),
        "instance_geometric_iou": float(
            instance["matched_geometric_iou"]
        ),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_abs_error": float(instance["count_abs_error"]),
    }


@torch.inference_mode()
def _run_models(
    args: argparse.Namespace,
    taxonomy: FloorplanTaxonomy,
    loader: DataLoader,
) -> list[dict]:
    device = torch.device(args.device)
    rows = []
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        outputs = model_forward(args, taxonomy, footprint, image, device)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        dense = room_ids[outputs["room_logits"].argmax(dim=1)] * footprint.long()
        wall_probability = outputs["wall_logits"].sigmoid().cpu().numpy()
        for index, sample_name in enumerate(names):
            labels = dense[index].cpu().numpy()
            mask = footprint[index].cpu().numpy()
            expected = target[index].cpu().numpy()
            for arm, threshold in _arms():
                mode = (
                    "none"
                    if arm == "dense"
                    else arm
                    if arm == "boundary_wall"
                    else "predicted_wall"
                    if arm.startswith("predicted_wall")
                    else arm.split("_")[0]
                )
                fused = fuse_dense_room_walls(
                    labels,
                    mask,
                    wall_probability[index],
                    taxonomy,
                    mode=mode,
                    wall_threshold=(
                        0.5 if threshold is None else float(threshold)
                    ),
                )
                rows.append(
                    {
                        "sample_key": sample_name,
                        "arm": arm,
                        **_room_metrics(
                            fused.labels,
                            expected,
                            mask,
                            taxonomy,
                        ),
                        "wall_pixels": fused.wall_pixels,
                        "boundary_pixels": fused.boundary_pixels,
                    }
                )
    return rows


def model_forward(
    args: argparse.Namespace,
    taxonomy: FloorplanTaxonomy,
    footprint: torch.Tensor,
    image: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not hasattr(model_forward, "model"):
        model_forward.model = load_raster_control_bank(
            args.checkpoint,
            taxonomy,
            device,
        )
    return model_forward.model(
        footprint,
        control_image=image,
    )


@torch.inference_mode()
def _structural_selected(
    args: argparse.Namespace,
    taxonomy: FloorplanTaxonomy,
    loader: DataLoader,
    selected_arm: str,
) -> dict[str, float]:
    device = torch.device(args.device)
    rows = []
    threshold = (
        None
        if selected_arm in {"dense", "boundary_wall"}
        else float(selected_arm.rsplit("_", 1)[-1])
    )
    mode = (
        "none"
        if selected_arm == "dense"
        else selected_arm
        if selected_arm == "boundary_wall"
        else "predicted_wall"
        if selected_arm.startswith("predicted_wall")
        else selected_arm.split("_")[0]
    )
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        outputs = model_forward(args, taxonomy, footprint, image, device)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        dense = room_ids[outputs["room_logits"].argmax(dim=1)] * footprint.long()
        wall_probability = outputs["wall_logits"].sigmoid().cpu().numpy()
        for index in range(target.shape[0]):
            fused = fuse_dense_room_walls(
                dense[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                wall_probability[index],
                taxonomy,
                mode=mode,
                wall_threshold=0.5 if threshold is None else threshold,
            )
            metrics = generation_plausibility(
                fused.labels,
                footprint[index].cpu().numpy(),
                taxonomy,
                explicit_rooms=True,
            )
            rows.append(
                {
                    "validity": metrics["validity"],
                    "wall_continuity": metrics["wall_continuity_rate"],
                    "functional_watertight": metrics["functional_watertight"],
                    "plan_plausibility": float(metrics["plan_plausible"]),
                    "room_overlap": metrics["room_overlap_ratio"],
                    "out_of_boundary": metrics["out_of_boundary_ratio"],
                }
            )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    taxonomy_payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(taxonomy_payload["taxonomy"])
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=int(args.offset) + int(args.limit),
        image_mode="native",
    )
    subset = Subset(
        dataset,
        list(range(int(args.offset), int(args.offset) + int(args.limit))),
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    rows = _run_models(args, taxonomy, loader)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    metrics = {
        arm: {
            key: _summary([row[key] for row in arm_rows])
            for key in (
                "room_macro_iou",
                "instance_geometric_iou",
                "instance_f1",
                "type_accuracy",
                "count_abs_error",
                "wall_pixels",
                "boundary_pixels",
            )
        }
        for arm, arm_rows in grouped.items()
    }
    baseline = metrics["dense"]
    ranking = sorted(
        metrics,
        key=lambda arm: (
            metrics[arm]["room_macro_iou"]["mean"]
            + 0.25 * metrics[arm]["instance_f1"]["mean"]
            + 0.1 * metrics[arm]["type_accuracy"]["mean"]
        ),
        reverse=True,
    )
    selected_arm = ranking[0]
    structural = _structural_selected(
        args,
        taxonomy,
        loader,
        selected_arm,
    )
    report = {
        "schema_version": "dense-room-wall-fusion-v1",
        "scope": "dense room detection plus predicted wall instance fusion",
        "split": "val",
        "offset": args.offset,
        "cases": args.limit,
        "target_used_at_inference": False,
        "ranking_rule": "Room IoU + 0.25 * instance F1 + 0.1 * type accuracy",
        "selected_arm": selected_arm,
        "metrics": metrics,
        "deltas_vs_dense": {
            arm: {
                key: metrics[arm][key]["mean"] - baseline[key]["mean"]
                for key in (
                    "room_macro_iou",
                    "instance_f1",
                    "type_accuracy",
                    "count_abs_error",
                )
            }
            for arm in ranking
        },
        "selected_structural": structural,
        "case_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
