#!/usr/bin/env python3
"""Test whether query-mask support improves the deployed room-label contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_dense_query_blend_candidate_v1 import case_metrics

from floorflow.data.control_contract import sha256_file
from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.heads.dense_query_support_gate import (
    apply_room_support_gate,
    query_mask_support_probability,
)
from floorflow.heads.raster_control_bank import (
    RasterDenseQueryRoomBlend,
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
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--query-weight", type=float, default=None)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    parser.add_argument(
        "--confidence-preserve",
        nargs="+",
        type=float,
        default=[0.7, 0.8, 0.9],
        help="Retain dense room support above each confidence level.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    )
    blend_model = RasterDenseQueryRoomBlend(
        dense_model,
        query_model,
        args.query_weight,
    ).eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    rows: dict[str, list[dict]] = {}
    for target, footprint, image, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        footprint_room = footprint[:, None] if footprint.ndim == 3 else footprint
        blend_logits, dense_logits = blend_model.forward_with_dense(
            footprint,
            control_image=image,
        )
        query_output = query_model(footprint, control_image=image)
        class_probabilities = {
            "dense": dense_logits.softmax(dim=1),
            "blend50": blend_logits.softmax(dim=1),
        }
        support_probabilities = {
            mode: query_mask_support_probability(
                query_output["mask_logits"],
                mode=mode,
            )
            for mode in ("max", "union")
        }
        dense_confidence = class_probabilities["dense"].amax(dim=1)
        for confidence in args.confidence_preserve:
            support_probabilities[
                f"union_or_conf{confidence:.2f}"
            ] = torch.maximum(
                support_probabilities["union"],
                (dense_confidence >= confidence).float(),
            )
        labels_by_arm = {}
        for class_name, probability in class_probabilities.items():
            base = room_ids[probability.argmax(dim=1)] * footprint_room[:, 0]
            labels_by_arm[f"{class_name}_footprint"] = base
            for support_name, support in support_probabilities.items():
                for threshold in args.thresholds:
                    name = f"{class_name}_{support_name}_{threshold:.2f}"
                    labels_by_arm[name] = apply_room_support_gate(
                        base,
                        support,
                        footprint_room,
                        threshold=threshold,
                    )
        for index, case_name in enumerate(names):
            for arm, labels in labels_by_arm.items():
                rows.setdefault(arm, []).append(
                    {
                        "case": str(case_name),
                        **case_metrics(
                            labels[index].cpu().numpy(),
                            target[index].cpu().numpy(),
                            footprint_room[index, 0].cpu().numpy(),
                            taxonomy,
                        ),
                    }
                )
    summary = {
        arm: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
            if key != "case"
        }
        for arm, values in rows.items()
    }
    report = {
        "schema_version": "dense-query-support-diagnostic-v1",
        "protocol": "predicted_control",
        "target_used_at_inference": False,
        "test_opened": False,
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "config": {
            "dense_checkpoint": str(args.dense_checkpoint),
            "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
            "query_checkpoint": str(args.query_checkpoint),
            "query_checkpoint_sha256": sha256_file(args.query_checkpoint),
            "taxonomy_sha256": sha256_file(args.cache_root / "taxonomy.json"),
            "query_weight": args.query_weight,
            "query_weight_source": query_weight_source,
            "thresholds": args.thresholds,
        },
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ranked = sorted(
        summary.items(),
        key=lambda item: (
            item[1]["room_iou"]
            + 0.75 * item[1]["instance_f1"]
            - 0.02 * item[1]["count_mae"]
        ),
        reverse=True,
    )
    print(json.dumps({"top_arms": ranked[:10]}, indent=2))


if __name__ == "__main__":
    main()
