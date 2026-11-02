#!/usr/bin/env python3
"""Evaluate dense room partition combined with wall/object semantics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.geometry.core import compose_known_exterior
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.inference.retrieval_prior import ContourRoomRetriever, prior_one_hot
from floorflow.heads.room_partition import RoomPartitionHead, compose_partition_with_semantics
from floorflow.inference.sampling_policy import SamplingPolicy
from floorflow.data.taxonomy import FloorplanTaxonomy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-checkpoint", type=Path, required=True)
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "outputs/cache/cubicasa5k_full_v1")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / ".data/cubicasa5k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/eval/room_partition/metrics.json")
    args = parser.parse_args()
    payload = torch.load(args.partition_checkpoint, map_location="cpu", weights_only=False)
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    backbone, config, _, _ = load_checkpoint_model(args.semantic_checkpoint, args.device, taxonomy=taxonomy)
    partition_config = payload.get("config", {})
    retrieval_top_k = int(partition_config.get("retrieval_top_k", 0))
    retriever = None
    if retrieval_top_k > 0:
        train = CubiCasaFloorplanDataset(
            args.data_root,
            "train",
            image_size=int(config.get("image_size", 256)),
            cache_root=args.cache_root,
            preload=True,
        )
        retriever = ContourRoomRetriever(train._targets, train._footprints, taxonomy, retrieval_top_k)
    head = RoomPartitionHead(
        backbone.output.in_channels,
        len(taxonomy.room_ids),
        prior_channels=len(taxonomy.room_ids) if retriever is not None else 0,
    ).to(args.device)
    head.load_state_dict(payload["head"], strict=False)
    head.eval()
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
        cache_root=args.cache_root,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.no_grad():
        for target, footprint, floor_ids, _names in loader:
            target, footprint, floor_ids = target.to(args.device), footprint.to(args.device), floor_ids.to(args.device)
            zero = torch.zeros(target.shape[0], backbone.target_channels, *target.shape[-2:], device=args.device)
            features = backbone._encode_features(
                zero,
                torch.zeros(target.shape[0], device=args.device),
                footprint,
                floor_ids=floor_ids,
            )
            prior = None
            if retriever is not None:
                prior = prior_one_hot(
                    retriever.retrieve(footprint.cpu().numpy()),
                    len(taxonomy.room_ids),
                    args.device,
                )
            room_logits = head(features, prior)
            semantic_logits = backbone.semantic_endpoint(features)
            semantic = torch.cat([torch.zeros_like(semantic_logits[:, :1]), semantic_logits], dim=1).argmax(dim=1)
            labels = compose_partition_with_semantics(room_logits, semantic, footprint, taxonomy)
            labels = compose_known_exterior(labels, footprint, opening_ids=taxonomy.opening_ids)
            for index in range(target.shape[0]):
                rows.append(
                    floorplan_metrics(
                        labels[index : index + 1],
                        target[index : index + 1],
                        footprint[index : index + 1],
                        taxonomy,
                    )
                )
    keys = (
        "room_macro_iou",
        "room_area_coverage",
        "room_component_gap",
        "interior_wall_iou",
        "opening_f1",
        "furniture_iou",
        "furniture_room_containment",
    )
    report = {
        "cases": len(dataset),
        "claim_role": "dense_partition_plus_semantic_overlay",
        "best_of_n_used": False,
        "sampling_policy": SamplingPolicy(purpose="single_sample").manifest(),
        "partition_checkpoint": str(args.partition_checkpoint),
        "semantic_checkpoint": str(args.semantic_checkpoint),
        "metrics": {key: float(np.mean([row[key] for row in rows])) for key in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
