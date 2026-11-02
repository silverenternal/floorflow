#!/usr/bin/env python3
"""Evaluate a floorflow checkpoint on the official skv4 plan-level test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from floorflow.data.skv4 import rasterize_skv4_plan
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.training.checkpoint import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/cubicasa5k_full_v2"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--retrieval-prior-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument("--retrieval-metric", default="mask_iou")
    parser.add_argument(
        "--output-mode",
        choices=("auto", "base", "distribution"),
        default="auto",
    )
    parser.add_argument("--room-logit-adapter", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    distribution_mode = (
        args.output_mode == "distribution"
        or (args.output_mode == "auto" and "pixel_head" in payload)
    )
    if distribution_mode:
        model, taxonomy, _payload = restore_distribution_model(
            args.checkpoint,
            args.device,
        )
    else:
        model, _config, _taxonomy, _state = load_checkpoint_model(
            args.checkpoint,
            args.device,
            taxonomy=taxonomy,
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
    adapter = (
        json.loads(args.room_logit_adapter.read_text(encoding="utf-8"))
        if args.room_logit_adapter is not None
        else None
    )
    rows = []
    with (args.skv4_root / "test.jsonl").open(encoding="utf-8") as source:
        for line in source:
            plan = json.loads(line)
            target, footprint = rasterize_skv4_plan(
                plan,
                taxonomy,
                args.image_size,
            )
            footprint_t = torch.from_numpy(footprint)[None, None].float().to(args.device)
            if distribution_mode:
                distribution = model.contour_initial_state(
                    footprint_t,
                    floor_ids=torch.zeros(1, dtype=torch.long, device=args.device),
                )
                logits = distribution["pixel"]["room_logits"]
            else:
                zeros = torch.zeros(
                    1,
                    model.target_channels,
                    args.image_size,
                    args.image_size,
                    device=args.device,
                )
                _velocity, auxiliary = model.forward_with_aux(
                    zeros,
                    torch.zeros(1, device=args.device),
                    footprint_t,
                )
                logits = auxiliary["room_partition_logits"]
            if adapter is not None:
                if list(adapter["class_ids"]) != list(taxonomy.room_ids):
                    raise ValueError("room logit adapter class ids do not match")
                bias = torch.tensor(
                    adapter["bias"],
                    device=args.device,
                )[None, :, None, None]
                logits = (
                    logits + bias
                ) / max(0.05, float(adapter["temperature"]))
            if memory is not None:
                prior = memory.retrieve_logits(
                    np.asarray(footprint, dtype=np.float32)[None],
                    taxonomy,
                    args.device,
                    top_k=args.retrieval_top_k,
                )
                logits = logits + float(args.retrieval_prior_weight) * prior
            room_ids = torch.tensor(taxonomy.room_ids, device=args.device)
            prediction = room_ids[logits.argmax(dim=1)][0].cpu().numpy()
            metrics = room_metric_suite(
                prediction,
                target,
                footprint,
                taxonomy,
            )
            rows.append(
                {
                    "id": plan["id"],
                    "room_macro_iou": metrics["visible_semantic"]["room_macro_iou"],
                    **metrics["matched_instance"],
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
    report = {
        "dataset": "skv4",
        "split": "official_plan_test",
        "case_count": len(rows),
        "checkpoint": str(args.checkpoint),
        "output_mode": "distribution_pixel" if distribution_mode else "base_partition",
        "room_logit_adapter": (
            str(args.room_logit_adapter)
            if args.room_logit_adapter is not None
            else None
        ),
        "retrieval_prior_weight": float(args.retrieval_prior_weight),
        "retrieval_top_k": int(args.retrieval_top_k),
        "retrieval_metric": args.retrieval_metric,
        "summary": summary,
        "rows": rows,
        "best_of_n_used": False,
        "human_rating": "not reported",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
