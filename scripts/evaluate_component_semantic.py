#!/usr/bin/env python3
"""Apply a learned component classifier without changing room support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.component_semantic import (
    ComponentSemanticClassifier,
    assign_component_classes,
    component_records,
)
from floorflow.objectives.condition_objective import partition_room_labels
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--offset", type=int, default=191)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, config, taxonomy, _ = load_checkpoint_model(
        args.checkpoint,
        device,
    )
    model.eval().requires_grad_(False)
    payload = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classifier_config = payload.get("config", {})
    classifier = ComponentSemanticClassifier(
        int(payload["feature_size"]),
        int(payload["num_classes"]),
        hidden_size=int(classifier_config.get("hidden_size", 128)),
        dropout=float(classifier_config.get("dropout", 0.1)),
    ).to(device)
    classifier.load_state_dict(payload["model"])
    classifier.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
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
    memory = ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
    prior_weight = float(
        payload.get("config", {}).get("retrieval_prior_weight", 0.5)
    )
    rows = []
    for target, footprint, floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
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
            floor_ids=floor_ids.to(device),
        )
        room_logits = aux["room_partition_logits"]
        retrieval = memory.retrieve_logits(
            footprint.cpu().numpy(),
            taxonomy,
            device,
        )
        labels = (
            partition_room_labels(
                room_logits + prior_weight * retrieval,
                taxonomy,
            )
            * footprint.long()
        )
        model_probability = room_logits.softmax(dim=1).cpu().numpy()
        retrieval_probability = retrieval.softmax(dim=1).cpu().numpy()
        for index in range(target.shape[0]):
            features, _targets, component_ids = component_records(
                labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                model_probability[index],
                retrieval_probability[index],
                footprint[index].cpu().numpy(),
                taxonomy.room_ids,
            )
            if len(features):
                probability = (
                    classifier(torch.from_numpy(features).to(device))
                    .softmax(dim=-1)
                    .cpu()
                    .numpy()
                )
            else:
                probability = np.zeros((0, len(taxonomy.room_ids)))
            prediction = assign_component_classes(
                labels[index].cpu().numpy(),
                probability,
                component_ids,
                taxonomy.room_ids,
            )
            prediction_t = torch.from_numpy(prediction).to(device)
            room_ids = torch.tensor(taxonomy.room_ids, device=device)
            suite = room_metric_suite(
                prediction,
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(
                    prediction_t, room_ids
                ).cpu().numpy(),
                predicted_instance_support=torch.isin(
                    prediction_t, room_ids
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[index], room_ids
                ).cpu().numpy(),
            )
            instance = suite["matched_instance"]
            rows.append(
                {
                    "room_macro_iou": suite["visible_semantic"][
                        "room_macro_iou"
                    ],
                    "instance_geometric_iou": instance[
                        "matched_geometric_iou"
                    ],
                    "instance_room_f1": instance[
                        "instance_f1_at_threshold"
                    ],
                    "instance_type_accuracy": instance[
                        "matched_type_accuracy"
                    ],
                    "class_constrained_iou": instance[
                        "matched_class_constrained_iou"
                    ],
                }
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "classifier": str(args.classifier),
        "split": args.split,
        "cases": len(rows),
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
