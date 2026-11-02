#!/usr/bin/env python3
"""Evaluate a checkpoint's room-partition branch without structural overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.room_mask_decoder import fused_room_mask_logits
from floorflow.inference.instance_semantics import (
    consolidate_room_instances,
    consolidate_room_instances_with_prior,
    enforce_room_instance_count,
    graph_refine_room_instances,
)
from floorflow.objectives.condition_objective import partition_room_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.representation.core import extract_room_instances
from floorflow.representation.program_conditioning import apply_program_constraint
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
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=191)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--room-prior-weights-report", type=Path, default=None)
    parser.add_argument("--instance-semantic-vote", action="store_true")
    parser.add_argument("--instance-prior-vote", action="store_true")
    parser.add_argument("--instance-prior-weights-report", type=Path, default=None)
    parser.add_argument("--graph-room-context", action="store_true")
    parser.add_argument("--graph-neighbor-weight", type=float, default=0.2)
    parser.add_argument("--graph-steps", type=int, default=2)
    parser.add_argument("--graph-margin-threshold", type=float, default=None)
    parser.add_argument("--class-pair-prior", type=Path, default=None)
    parser.add_argument("--graph-class-strength", type=float, default=0.0)
    parser.add_argument("--graph-consensus-threshold", type=float, default=None)
    parser.add_argument("--graph-pair-margin-threshold", type=float, default=None)
    parser.add_argument("--enforce-room-count", action="store_true")
    parser.add_argument("--room-count-buffer", type=int, default=0)
    parser.add_argument("--retrieval-program-penalty", type=float, default=0.0)
    parser.add_argument("--use-room-mask-logits", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _state = load_checkpoint_model(
        args.checkpoint,
        device,
    )
    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
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
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
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
            dtype=torch.float32,
        )
    instance_class_weights = class_weights
    if args.instance_prior_weights_report is not None:
        instance_class_weights = torch.tensor(
            json.loads(
                args.instance_prior_weights_report.read_text(encoding="utf-8")
            )["weights"],
            device=device,
            dtype=torch.float32,
        )
    class_pair_prior = None
    if args.class_pair_prior is not None:
        class_pair_prior = torch.tensor(
            json.loads(
                args.class_pair_prior.read_text(encoding="utf-8")
            )["conditional"],
            device=device,
            dtype=torch.float32,
        )
    rows: list[dict[str, float]] = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        zeros = torch.zeros(
            target.shape[0],
            model.target_channels,
            *target.shape[-2:],
            device=device,
        )
        _velocity, aux = model.forward_with_aux(
            zeros,
            torch.zeros(target.shape[0], device=device),
            footprint,
            floor_ids=floor_ids,
        )
        room_logits = aux["room_partition_logits"]
        if (
            args.use_room_mask_logits
            and aux.get("room_mask_predictions") is not None
        ):
            room_logits = fused_room_mask_logits(
                aux["room_mask_predictions"],
                len(taxonomy.room_ids),
            )
        retrieval_prior = None
        raw_retrieval_prior = None
        if memory is not None:
            raw_retrieval_prior = memory.retrieve_logits(
                footprint.cpu().numpy(),
                taxonomy,
                device,
            )
            retrieval_prior = raw_retrieval_prior
            if class_weights is not None:
                retrieval_prior = (
                    retrieval_prior
                    * class_weights[None, :, None, None]
                )
            room_logits = (
                room_logits
                + float(args.retrieval_prior_weight) * retrieval_prior
            )
            if args.retrieval_program_penalty > 0:
                retrieval_labels = memory.retrieve_labels(
                    footprint.cpu().numpy(),
                    top_k=1,
                ).to(device)
                program = torch.stack(
                    [
                        (retrieval_labels == class_id).any(dim=(1, 2))
                        for class_id in taxonomy.room_ids
                    ],
                    dim=1,
                ).float()
                room_logits = apply_program_constraint(
                    room_logits,
                    program,
                    absent_penalty=args.retrieval_program_penalty,
                )
        prediction = (
            partition_room_labels(room_logits, taxonomy)
            * footprint.long()
        )
        if args.instance_semantic_vote:
            prediction = consolidate_room_instances(
                prediction,
                room_logits,
                footprint,
                taxonomy.room_ids,
            )
        if args.graph_room_context:
            if raw_retrieval_prior is None or instance_class_weights is None:
                raise ValueError(
                    "graph context requires retrieval and instance weights"
                )
            prediction = graph_refine_room_instances(
                prediction,
                aux["room_partition_logits"],
                raw_retrieval_prior,
                footprint,
                taxonomy.room_ids,
                instance_class_weights,
                neighbor_weight=args.graph_neighbor_weight,
                steps=args.graph_steps,
                margin_threshold=args.graph_margin_threshold,
                class_pair_prior=class_pair_prior,
                class_strength=args.graph_class_strength,
                neighbor_consensus_threshold=args.graph_consensus_threshold,
                pair_margin_threshold=args.graph_pair_margin_threshold,
            )
        elif args.instance_prior_vote:
            if raw_retrieval_prior is None or instance_class_weights is None:
                raise ValueError(
                    "instance-prior vote requires retrieval and class weights"
                )
            prediction = consolidate_room_instances_with_prior(
                prediction,
                aux["room_partition_logits"],
                raw_retrieval_prior,
                footprint,
                taxonomy.room_ids,
                instance_class_weights,
            )
        if args.enforce_room_count:
            if memory is None:
                raise ValueError("room-count prior requires retrieval memory")
            retrieval_labels = memory.retrieve_labels(
                footprint.cpu().numpy(),
                top_k=1,
            ).to(device)
            for index in range(target.shape[0]):
                target_count = len(
                    extract_room_instances(
                        retrieval_labels[index].cpu().numpy(),
                        taxonomy,
                    )
                )
                prediction[index : index + 1] = enforce_room_instance_count(
                    prediction[index : index + 1],
                    aux["room_partition_logits"][index : index + 1],
                    footprint[index : index + 1],
                    taxonomy.room_ids,
                    target_count + int(args.room_count_buffer),
                )
        for index in range(target.shape[0]):
            predicted_support = torch.isin(prediction[index], room_ids)
            target_support = torch.isin(target[index], room_ids)
            suite = room_metric_suite(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_support.cpu().numpy(),
                predicted_instance_support=predicted_support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
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
        "cases": len(dataset),
        "best_of_n_used": False,
        "metrics": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
