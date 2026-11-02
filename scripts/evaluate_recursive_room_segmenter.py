#!/usr/bin/env python3
"""Compare direct and learned-recursive room segmenter decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.inference.recursive_room_prior import (
    load_recursion_policy,
    predict_recursive_room_prior,
)
from floorflow.inference.room_segmenter_prior import load_room_segmenter_prior
from floorflow.recursion.complexity import (
    POLICY_ACTION_NAMES,
    complexity_features,
    complexity_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-segmenter-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=Path("outputs/train/recursion_policy_v1/best.pt"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf-pixels", type=int, default=256)
    parser.add_argument("--context-fraction", type=float, default=0.25)
    parser.add_argument("--recursive-blend", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def room_metrics(prediction, target, footprint, taxonomy):
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
        "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
        "room_support_iou": suite["occlusion_aware_support"]["room_support_iou"],
        "instance_geometric_iou": instance["matched_geometric_iou"],
        "instance_room_f1": instance["instance_f1_at_threshold"],
        "instance_type_accuracy": instance["matched_type_accuracy"],
        "class_constrained_iou": instance[
            "matched_class_constrained_iou"
        ],
    }


def summarize(rows, key):
    return {
        metric: float(np.mean([row[key][metric] for row in rows]))
        for metric in rows[0][key]
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = min(len(dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    segmenter, _config = load_room_segmenter_prior(
        args.room_segmenter_checkpoint,
        dataset.taxonomy,
        device,
    )
    policy = load_recursion_policy(args.policy_checkpoint, device)
    rows = []
    for target, footprint, _floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        direct_logits = segmenter(footprint)
        recursive = predict_recursive_room_prior(
            policy,
            segmenter,
            footprint,
            max_depth=args.max_depth,
            min_leaf_pixels=args.min_leaf_pixels,
            context_fraction=args.context_fraction,
            global_logits=direct_logits,
            recursive_blend=args.recursive_blend,
        )
        target_features = complexity_features(
            footprint,
            target,
            dataset.taxonomy,
        )
        complexity = complexity_score(target_features).cpu()
        room_ids = torch.tensor(dataset.taxonomy.room_ids, device=device)
        direct_labels = room_ids[direct_logits.argmax(dim=1)] * footprint[:, 0]
        recursive_labels = (
            room_ids[recursive.logits.argmax(dim=1)] * footprint[:, 0]
        )
        hybrid_labels = torch.where(
            (recursive.action == 0)[:, None, None],
            direct_labels,
            recursive_labels,
        )
        for index, name in enumerate(names):
            values = {
                "direct": room_metrics(
                    direct_labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index, 0].cpu().numpy(),
                    dataset.taxonomy,
                ),
                "recursive": room_metrics(
                    recursive_labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index, 0].cpu().numpy(),
                    dataset.taxonomy,
                ),
                "hybrid": room_metrics(
                    hybrid_labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index, 0].cpu().numpy(),
                    dataset.taxonomy,
                ),
            }
            regions = recursive.regions[index]
            rows.append(
                {
                    "name": name,
                    "complexity": float(complexity[index]),
                    "action": POLICY_ACTION_NAMES[int(recursive.action[index])],
                    "recursion_strength": float(
                        recursive.recursion_strength[index]
                    ),
                    "regions": len(regions),
                    "max_depth": max(region.depth for region in regions),
                    **values,
                }
            )
        print(json.dumps({"cases": len(rows)}), flush=True)
    summary = {
        "split": args.split,
        "offset": args.offset,
        "cases": len(rows),
        "best_of_n_used": False,
        "direct": summarize(rows, "direct"),
        "recursive": summarize(rows, "recursive"),
        "hybrid": summarize(rows, "hybrid"),
        "actions": {
            action: sum(row["action"] == action for row in rows)
            for action in POLICY_ACTION_NAMES
        },
        "depth_audit": {
            str(depth): {
                "cases": sum(row["max_depth"] == depth for row in rows),
                "room_macro_iou": float(
                    np.mean(
                        [
                            row["hybrid"]["room_macro_iou"]
                            for row in rows
                            if row["max_depth"] == depth
                        ]
                    )
                )
                if any(row["max_depth"] == depth for row in rows)
                else None,
            }
            for depth in range(args.max_depth + 1)
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "cases",
                    "direct",
                    "recursive",
                    "hybrid",
                    "actions",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
