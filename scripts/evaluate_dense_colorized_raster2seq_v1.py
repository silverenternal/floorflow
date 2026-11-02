#!/usr/bin/env python3
"""Evaluate target-free dense-to-Raster2Seq colorized routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.data.raster_control import load_cubicasa_rgb_raster
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.raster2seq_external import (
    decode_raster2seq_sequence,
    render_raster2seq_predictions,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import load_raster_control_bank
from floorflow.inference.dense_colorized_raster2seq import (
    dense_colorized_raster2seq_input,
)
from run_fp_raster2seq_matched_edits import load_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/raster_full_cad_bank_frozen_room_screen_v4/best.pt"
        ),
    )
    parser.add_argument(
        "--raster2seq-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/external_baselines/raster2seq/cubicasa5k/checkpoint.pth"
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "outputs/reports/dense_colorized_raster2seq_cal64_v1.json"
        ),
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


def _to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.stack(images))
        .permute(0, 3, 1, 2)
        .to(device)
    )
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    return tensor.float()


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
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
        "instance_geometric_iou": float(
            instance["matched_geometric_iou"]
        ),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_abs_error": float(instance["count_abs_error"]),
    }


def _raster2seq_decode(
    model,
    images: list[np.ndarray],
    taxonomy: FloorplanTaxonomy,
    image_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    with torch.inference_mode():
        predictions = model.forward_inference(
            _to_tensor(images, device),
            use_cache=True,
        )
    decoded = []
    for index in range(len(images)):
        polygons, labels = decode_raster2seq_sequence(
            predictions["gen_out"][index],
            predictions["pred_room_logits"][index],
            taxonomy,
            image_size=image_size,
        )
        decoded.append(
            render_raster2seq_predictions(
                polygons,
                labels,
                (256, 256),
                taxonomy,
            )
        )
    return decoded


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dense_payload = torch.load(
        args.dense_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    taxonomy = FloorplanTaxonomy.from_json(dense_payload["taxonomy"])
    dense_model = load_raster_control_bank(
        args.dense_checkpoint,
        taxonomy,
        device,
    )
    raster_model, image_size = load_model(
        args.raster2seq_checkpoint,
        device,
    )
    if image_size != 256:
        raise ValueError("expected Raster2Seq image size 256")
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        limit=int(args.offset) + int(args.limit),
        image_mode="native",
    )
    sample_map = {
        f"{sample.relative_case}__{sample.floor_id}": sample
        for sample in dataset.samples
    }
    loader = DataLoader(
        Subset(
            dataset,
            list(range(int(args.offset), int(args.offset) + int(args.limit))),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    arms = (
        "dense",
        "raster2seq_raw",
        "raster2seq_rooms",
        "raster2seq_rooms_walls",
        "raster2seq_full_structure",
    )
    rows = {arm: [] for arm in arms}
    case_rows = []
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        outputs = dense_model(footprint, control_image=image)
        room_ids = torch.tensor(taxonomy.room_ids, device=device)
        dense = room_ids[outputs["room_logits"].argmax(dim=1)] * footprint.long()
        raw_images = []
        colorized = {}
        for index, sample_name in enumerate(names):
            sample = sample_map[sample_name]
            raw_images.append(
                load_cubicasa_rgb_raster(sample.raster_path, image_size=256)
            )
            single_outputs = {
                key: value[index : index + 1]
                for key, value in outputs.items()
            }
            single_footprint = footprint[index : index + 1]
            colorized[sample_name] = {
                mode: dense_colorized_raster2seq_input(
                    single_outputs,
                    single_footprint,
                    taxonomy,
                    mode=mode,
                ).image
                for mode in ("rooms", "rooms_walls", "full_structure")
            }
        arm_images = {
            "raster2seq_raw": raw_images,
            "raster2seq_rooms": [
                colorized[name]["rooms"] for name in names
            ],
            "raster2seq_rooms_walls": [
                colorized[name]["rooms_walls"] for name in names
            ],
            "raster2seq_full_structure": [
                colorized[name]["full_structure"] for name in names
            ],
        }
        decoded = {
            arm: _raster2seq_decode(
                raster_model,
                images,
                taxonomy,
                image_size,
                device,
            )
            for arm, images in arm_images.items()
        }
        for index, sample_name in enumerate(names):
            expected = target[index].cpu().numpy()
            mask = footprint[index].cpu().numpy()
            arm_labels = {
                "dense": dense[index].cpu().numpy(),
                **{
                    arm: values[index]
                    for arm, values in decoded.items()
                },
            }
            for arm in arms:
                metrics = _room_metrics(
                    arm_labels[arm],
                    expected,
                    mask,
                    taxonomy,
                )
                rows[arm].append(metrics)
                case_rows.append(
                    {
                        "sample_key": sample_name,
                        "arm": arm,
                        **metrics,
                    }
                )
    metrics = {
        arm: {
            key: _summary([row[key] for row in arm_rows])
            for key in arm_rows[0]
        }
        for arm, arm_rows in rows.items()
    }
    ranking = sorted(
        arms,
        key=lambda arm: (
            metrics[arm]["room_macro_iou"]["mean"]
            + 0.25 * metrics[arm]["instance_f1"]["mean"]
            + 0.1 * metrics[arm]["type_accuracy"]["mean"]
        ),
        reverse=True,
    )
    selected_arm = ranking[0]
    selected_labels = []
    for row in case_rows:
        if row["arm"] == selected_arm:
            selected_labels.append(row)
    report = {
        "schema_version": "dense-colorized-raster2seq-v1",
        "scope": "target-free dense prediction routed into Raster2Seq",
        "split": "val",
        "offset": args.offset,
        "cases": args.limit,
        "target_used_at_inference": False,
        "ranking_rule": "Room IoU + 0.25 * instance F1 + 0.1 * type accuracy",
        "selected_arm": selected_arm,
        "metrics": metrics,
        "deltas_vs_dense": {
            arm: {
                key: metrics[arm][key]["mean"]
                - metrics["dense"][key]["mean"]
                for key in (
                    "room_macro_iou",
                    "instance_f1",
                    "type_accuracy",
                    "count_abs_error",
                )
            }
            for arm in ranking
        },
        "selected_case_count": len(selected_labels),
        "case_rows": case_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_arm": selected_arm,
                "metrics": metrics,
                "deltas_vs_dense": report["deltas_vs_dense"],
            },
            indent=2,
        )
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()
