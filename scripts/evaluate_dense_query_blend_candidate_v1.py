#!/usr/bin/env python3
"""Evaluate the fixed dense-query blend with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.control_contract import sha256_file
from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.training.raster_instance_evaluation import resolve_query_blend

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/native_room_384_full_32ep_v1/best.pt",
    )
    parser.add_argument(
        "--query-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_query_fullres_softroute_screen256_v1/best.pt",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=491)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--query-weight", type=float, default=None)
    parser.add_argument("--owner-semantic-reassignment", action="store_true")
    parser.add_argument("--force-query-count-gate", action="store_true")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/dense_query_blend_val491_v1.json",
    )
    return parser.parse_args()


def case_metrics(
    labels: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    suite = room_metric_suite(
        labels,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(labels, room_ids),
        target_room_support=np.isin(target, room_ids),
    )
    instance = suite["matched_instance"]
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "instance_precision": float(instance["instance_precision_at_threshold"]),
        "instance_recall": float(instance["instance_recall_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_mae": float(instance["count_abs_error"]),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.query_weight, query_weight_source = resolve_query_blend(
        args.query_checkpoint,
        args.query_weight,
    )
    device = torch.device(args.device)
    dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=None,
        image_mode="native",
    )
    dataset.samples = dataset.samples[
        int(args.offset) : int(args.offset) + int(args.limit)
    ]
    taxonomy = dataset.taxonomy
    dense_model = load_raster_room_model(
        args.dense_checkpoint,
        taxonomy,
        device,
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
        owner_semantic_reassignment=(
            True if args.owner_semantic_reassignment else None
        ),
        query_count_gate=True if args.force_query_count_gate else None,
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    rows = {"dense": [], "blend": []}
    case_counter = 0
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        dense_probability = dense_model(
            footprint,
            control_image=image,
        ).softmax(dim=1)
        query_probability = query_model(
            footprint,
            control_image=image,
        )["fused_logits"].softmax(dim=1)
        blend_probability = (
            (1.0 - float(args.query_weight)) * dense_probability
            + float(args.query_weight) * query_probability
        )
        footprint_room = footprint[:, None] if footprint.ndim == 3 else footprint
        predictions = {
            "dense": (room_ids[dense_probability.argmax(dim=1)] * footprint_room[:, 0]),
            "blend": (room_ids[blend_probability.argmax(dim=1)] * footprint_room[:, 0]),
        }
        for index, name in enumerate(names):
            target_np = target[index].cpu().numpy()
            footprint_np = footprint_room[index, 0].cpu().numpy()
            for arm, labels in predictions.items():
                rows[arm].append(
                    {
                        "index": int(args.offset) + case_counter,
                        "case": str(name),
                        **case_metrics(
                            labels[index].cpu().numpy(),
                            target_np,
                            footprint_np,
                            taxonomy,
                        ),
                    }
                )
            case_counter += 1
    keys = [
        key
        for key in rows["dense"][0]
        if key not in {"index", "case"}
    ]
    stats = {
        key: paired_delta(
            rows["blend"],
            rows["dense"],
            key,
            iterations=args.iterations,
            seed=args.seed,
            lower_is_better=key in {
                "count_mae",
            },
        )
        for key in keys
    }
    report = {
        "schema_version": "dense-query-blend-candidate-v1",
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
        "test_opened": False,
        "config": {
            "dense_checkpoint": str(args.dense_checkpoint),
            "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
            "query_checkpoint": str(args.query_checkpoint),
            "query_checkpoint_sha256": sha256_file(args.query_checkpoint),
            "cache_root": str(args.cache_root),
            "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
            "validation_offset": int(args.offset),
            "validation_limit": int(args.limit),
            "image_size": int(args.image_size),
            "query_weight": float(args.query_weight),
            "query_weight_source": query_weight_source,
            "ci_method": "paired_percentile_bootstrap",
            "ci_iterations": int(args.iterations),
            "ci_seed": int(args.seed),
        },
        "query_weight": float(args.query_weight),
        "query_weight_source": query_weight_source,
        "validation_offset": int(args.offset),
        "validation_cases": len(dataset),
        "summary": {
            arm: {
                key: float(np.mean([row[key] for row in values]))
                for key in keys
            }
            for arm, values in rows.items()
        },
        "paired": stats,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": report["summary"], "paired": stats}, indent=2))


if __name__ == "__main__":
    main()
