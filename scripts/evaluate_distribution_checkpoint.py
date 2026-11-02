#!/usr/bin/env python3
"""Evaluate coupled pixel/element distribution checkpoints on room metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.inference.instance_semantics import (
    consolidate_room_instances_with_prior,
)
from floorflow.representation.contour_memory import ContourPartitionMemory


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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--room-prior-weights-report", type=Path, default=None)
    parser.add_argument("--instance-prior-vote", action="store_true")
    parser.add_argument("--instance-prior-weights-report", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = 41,
    iterations: int = 1000,
) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [
            data[generator.integers(0, len(data), len(data))].mean()
            for _ in range(iterations)
        ]
    )
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
    )
    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = min(len(full_dataset), args.offset + args.limit)
    dataset = Subset(full_dataset, range(args.offset, stop))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    memory = (
        ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
        if args.retrieval_prior_weight > 0
        else None
    )
    class_weights = None
    if args.room_prior_weights_report is not None:
        class_weights = torch.tensor(
            json.loads(
                args.room_prior_weights_report.read_text(encoding="utf-8")
            )["weights"],
            device=device,
        )
    instance_weights = class_weights
    if args.instance_prior_weights_report is not None:
        instance_weights = torch.tensor(
            json.loads(
                args.instance_prior_weights_report.read_text(encoding="utf-8")
            )["weights"],
            device=device,
        )
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    rows = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        distribution = model.contour_initial_state(
            footprint,
            floor_ids=floor_ids.to(device),
        )
        room_indices = torch.tensor(
            taxonomy.room_ids,
            device=device,
        )
        room_logits = distribution["pixel"]["logits"][:, room_indices]
        raw_prior = None
        if memory is not None:
            raw_prior = memory.retrieve_logits(
                footprint[:, 0].cpu().numpy(),
                taxonomy,
                device,
            )
            prior = raw_prior
            if class_weights is not None:
                prior = prior * class_weights[None, :, None, None]
            room_logits = (
                room_logits
                + args.retrieval_prior_weight * prior
            )
        prediction = room_ids[room_logits.argmax(dim=1)] * footprint[:, 0].long()
        if args.instance_prior_vote:
            if raw_prior is None or instance_weights is None:
                raise ValueError(
                    "instance prior vote requires retrieval and weights"
                )
            prediction = consolidate_room_instances_with_prior(
                prediction,
                distribution["pixel"]["room_logits"],
                raw_prior,
                footprint,
                taxonomy.room_ids,
                instance_weights,
            )
        for index in range(target.shape[0]):
            support = torch.isin(prediction[index], room_ids)
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index, 0].cpu().numpy(),
                taxonomy,
                predicted_room_support=support.cpu().numpy(),
                predicted_instance_support=support.cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "case": names[index],
                    "room_macro_iou": suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                    "class_constrained_iou": instance[
                        "matched_class_constrained_iou"
                    ],
                }
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "offset": args.offset,
        "cases": len(dataset),
        "condition_mode": "contour_only",
        "best_of_n_used": False,
        "retrieval_prior_weight": args.retrieval_prior_weight,
        "instance_prior_vote": args.instance_prior_vote,
        "metrics": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "case"
        },
        "bootstrap": {
            key: bootstrap_ci(
                [float(row[key]) for row in rows],
                seed=41,
            )
            for key in rows[0]
            if key != "case"
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
