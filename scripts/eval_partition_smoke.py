"""Evaluate raw room-partition outputs for small overfit checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.representation.contour_memory import ContourPartitionMemory
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
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument("--retrieval-metric", default="mask_iou")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        args.device,
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    memory = (
        ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=args.retrieval_top_k,
            metric=args.retrieval_metric,
        )
        if args.retrieval_prior_weight > 0
        else None
    )
    room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
    rows = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(args.device).long()
        footprint = footprint.to(args.device).float()
        floor_ids = floor_ids.to(args.device)
        zeros = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=args.device,
        )
        _velocity, auxiliary = model.forward_with_aux(
            zeros,
            torch.zeros(target.shape[0], device=args.device),
            footprint,
            floor_ids=floor_ids,
        )
        logits = auxiliary["room_partition_logits"]
        if memory is not None:
            prior = memory.retrieve_logits(
                footprint.detach().cpu().numpy(),
                taxonomy,
                args.device,
                top_k=args.retrieval_top_k,
            )
            logits = logits + float(args.retrieval_prior_weight) * prior
        partition = room_ids[logits.argmax(dim=1)]
        for index, name in enumerate(names):
            report = room_metric_suite(
                partition[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
            )
            rows.append(
                {
                    "name": name,
                    "room_macro_iou": report["visible_semantic"]["room_macro_iou"],
                    **report["matched_instance"],
                }
            )
    keys = (
        "room_macro_iou",
        "matched_geometric_iou",
        "instance_f1_at_threshold",
        "matched_type_accuracy",
        "matched_class_constrained_iou",
        "instance_class_constrained_f1_at_threshold",
    )
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "case_count": len(rows),
        "summary": summary,
        "rows": rows,
        "best_of_n_used": False,
        "retrieval_prior_weight": float(args.retrieval_prior_weight),
        "retrieval_top_k": int(args.retrieval_top_k),
        "retrieval_metric": args.retrieval_metric,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
