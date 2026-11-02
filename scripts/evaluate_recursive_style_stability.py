#!/usr/bin/env python3
"""Evaluate all-style quality and deterministic seed stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.distribution_overlay_metrics import room_metrics
from floorflow.evaluation.furniture_query_bridge import (
    apply_furniture_queries,
    load_furniture_query_head,
)
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.gated_room_decoupling import compose_gated_room_layers
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.recursive_room_prior import (
    load_graph_adapter,
    load_recursion_policy,
    predict_recursive_room_prior,
)
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.recursion.trust_gate import load_room_graph_trust_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--room-segmenter-checkpoint", type=Path, required=True)
    parser.add_argument("--recursion-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--recursive-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--recursive-graph-blend-scale", type=float, default=1.0)
    parser.add_argument("--room-graph-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--room-graph-trust-checkpoint", type=Path, default=None)
    parser.add_argument("--room-graph-trust-threshold", type=float, default=0.35)
    parser.add_argument("--furniture-query-checkpoint", type=Path, default=None)
    parser.add_argument("--furniture-query-threshold", type=float, default=0.7)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--styles", type=int, default=8)
    parser.add_argument("--room-prior-weight", type=float, default=3.0)
    parser.add_argument("--recursive-context-fraction", type=float, default=0.75)
    parser.add_argument("--recursive-blend", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def decode_style(
    model,
    memory,
    retrieval_weights,
    segmenter,
    policy,
    graph_adapter,
    trust_gate,
    furniture_query_head,
    targets,
    footprints,
    floor_ids,
    taxonomy,
    style_id,
    args,
):
    style_ids = torch.full(
        (footprints.shape[0],),
        int(style_id),
        dtype=torch.long,
        device=footprints.device,
    )
    direct_prior = predict_room_segmenter_prior(segmenter, footprints)
    recursive = predict_recursive_room_prior(
        policy,
        segmenter,
        footprints,
        max_depth=3,
        min_leaf_pixels=256,
        context_fraction=args.recursive_context_fraction,
        global_logits=direct_prior,
        recursive_blend=args.recursive_blend,
        graph_adapter=None,
    )
    room_prior = torch.where(
        (recursive.action == 0)[:, None, None, None],
        direct_prior,
        recursive.logits,
    )
    graph_prior = None
    if graph_adapter is not None:
        graph_recursive = predict_recursive_room_prior(
            policy,
            segmenter,
            footprints,
            max_depth=3,
            min_leaf_pixels=256,
            context_fraction=args.recursive_context_fraction,
            global_logits=direct_prior,
            recursive_blend=args.recursive_blend,
            graph_adapter=graph_adapter,
        )
        graph_prior = torch.where(
            (graph_recursive.action == 0)[:, None, None, None],
            direct_prior,
            graph_recursive.logits,
        )
    started = time.perf_counter()
    decoded = decode_overlay(
        model,
        memory,
        retrieval_weights,
        footprints,
        taxonomy,
        floor_ids=floor_ids,
        style_ids=style_ids,
        retrieval_prior_weight=0.0,
        room_prior_logits=room_prior,
        room_prior_weight=args.room_prior_weight,
        settings=DistributionOverlaySettings(
            opening_element_weight=0.0,
            furniture_element_weight=0.0,
        ),
    )
    runtime = time.perf_counter() - started
    layers = decoded["layers"]
    if graph_prior is not None:
        graph_decoded = decode_overlay(
            model,
            memory,
            retrieval_weights,
            footprints,
            taxonomy,
            floor_ids=floor_ids,
            style_ids=style_ids,
            retrieval_prior_weight=0.0,
            room_prior_logits=graph_prior,
            room_prior_weight=args.room_prior_weight,
            settings=DistributionOverlaySettings(
                opening_element_weight=0.0,
                furniture_element_weight=0.0,
            ),
        )
        layers, _diagnostics = compose_gated_room_layers(
            layers,
            graph_decoded["layers"],
            room_prior,
            graph_prior,
            footprints[:, 0],
            taxonomy,
            graph_confidence_threshold=args.room_graph_confidence_threshold,
            trust_gate=trust_gate,
            trust_threshold=args.room_graph_trust_threshold,
        )
    if furniture_query_head is not None:
        layers, _furniture_diagnostics = apply_furniture_queries(
            furniture_query_head,
            decoded["distribution"],
            layers,
            footprints,
            taxonomy,
            threshold=args.furniture_query_threshold,
        )
    labels = layers.visible_labels()
    rows = []
    for index in range(labels.shape[0]):
        rows.append(
            room_metrics(
                labels[index],
                targets[index],
                footprints[index, 0],
                layers.room_labels[index],
                taxonomy,
            )
        )
    return labels.cpu().numpy(), rows, runtime


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(args.checkpoint, device)
    segmenter, _ = load_room_segmenter_prior(
        args.room_segmenter_checkpoint,
        taxonomy,
        device,
    )
    policy = load_recursion_policy(args.recursion_policy_checkpoint, device)
    graph_adapter = (
        load_graph_adapter(
            args.recursive_graph_checkpoint,
            device,
            blend_scale=args.recursive_graph_blend_scale,
        )
        if args.recursive_graph_checkpoint is not None
        else None
    )
    trust_gate = (
        load_room_graph_trust_gate(args.room_graph_trust_checkpoint, device)
        if args.room_graph_trust_checkpoint is not None
        else None
    )
    furniture_query_head = None
    if args.furniture_query_checkpoint is not None:
        furniture_query_head, query_taxonomy = load_furniture_query_head(
            args.furniture_query_checkpoint,
            device,
        )
        if query_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("furniture query taxonomy does not match model")
    dataset = CubiCasaFloorplanDataset(
        ".data/cubicasa5k",
        args.split,
        image_size=256,
        cache_root="outputs/cache/cubicasa5k_full_v2",
        limit=args.limit,
        augment=False,
    )
    targets = []
    footprints = []
    floor_ids = []
    for index in range(len(dataset)):
        target, footprint, floor_id, _name = dataset[index]
        targets.append(target)
        footprints.append(footprint)
        floor_ids.append(floor_id)
    target = torch.stack(targets).to(device)
    footprint = torch.stack(footprints)[:, None].to(device)
    floor_id = torch.tensor(floor_ids, device=device)
    memory = None
    retrieval_weights = None
    outputs = {}
    style_rows = {}
    runtimes = {}
    for style_id in range(args.styles):
        labels, rows, runtime = decode_style(
            model,
            memory,
            retrieval_weights,
            segmenter,
            policy,
            graph_adapter,
            trust_gate,
            furniture_query_head,
            target,
            footprint,
            floor_id,
            taxonomy,
            style_id,
            args,
        )
        outputs[style_id] = labels
        style_rows[str(style_id)] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        runtimes[str(style_id)] = runtime
        print(json.dumps({"style_id": style_id, **style_rows[str(style_id)]}))
    repeated, _rows, _runtime = decode_style(
        model,
        memory,
        retrieval_weights,
        segmenter,
        policy,
        graph_adapter,
        trust_gate,
        furniture_query_head,
        target,
        footprint,
        floor_id,
        taxonomy,
        0,
        args,
    )
    seed_gap = float(np.mean(outputs[0] != repeated))
    room_support = np.isin(target.cpu().numpy(), taxonomy.room_ids)
    disagreements = []
    for left in range(args.styles):
        for right in range(left + 1, args.styles):
            disagreement = (
                (outputs[left] != outputs[right]) & room_support
            ).sum() / max(1, room_support.sum())
            disagreements.append(float(disagreement))
    room_values = [
        row["room_macro_iou"] for row in style_rows.values()
    ]
    runtime_values = list(runtimes.values())
    report = {
        "status": "complete",
        "split": args.split,
        "cases": len(dataset),
        "styles": args.styles,
        "best_of_n_used": False,
        "target_used_for_generation": False,
        "recursive_graph_checkpoint": (
            str(args.recursive_graph_checkpoint)
            if args.recursive_graph_checkpoint is not None
            else None
        ),
        "all_style_mean_room_iou": float(np.mean(room_values)),
        "worst_style_room_iou": float(np.min(room_values)),
        "style_separation": float(np.mean(disagreements)),
        "seed_iou_gap": seed_gap,
        "runtime_seconds_mean": float(np.mean(list(runtimes.values()))),
        "runtime_seconds_warm_mean": float(
            np.mean(sorted(runtime_values)[1:])
            if len(runtime_values) > 1
            else np.mean(runtime_values)
        ),
        "style_metrics": style_rows,
        "runtime_seconds": runtimes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
