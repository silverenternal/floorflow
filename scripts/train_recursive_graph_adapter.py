#!/usr/bin/env python3
"""Train a MiTA-routed room graph adapter on train-only region labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.dataset import CubiCasaFloorplanDataset, load_or_rasterize
from floorflow.heads.room_partition import build_room_partition_target
from floorflow.inference.recursive_room_prior import (
    decode_partition_region_logits,
    load_recursion_policy,
)
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.recursion.complexity import (
    POLICY_INPUT_FEATURE_NAMES,
    complexity_features,
)
from floorflow.recursion.graph import build_partition_graph
from floorflow.recursion.graph_adapter import (
    RecursiveSemanticGraphAdapter,
    region_node_features,
)
from floorflow.recursion.partition import (
    propose_partition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recursion-policy", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".data/cubicasa5k"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--limit-train", type=int, default=1024)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--router-count", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--blend", type=float, default=0.15)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/recursive_graph_adapter_v1"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/recursive_graph_adapter_v1.json"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def room_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    values = []
    for class_index in range(int(target.max(initial=-1)) + 1):
        selected = target == class_index
        if not selected.any():
            continue
        predicted = prediction == class_index
        values.append(
            float((predicted & selected).sum())
            / max(1.0, float((predicted | selected).sum()))
        )
    return float(np.mean(values)) if values else 0.0


@torch.inference_mode()
def prepare_case(
    sample,
    dataset,
    segmenter,
    policy,
    device,
    image_size,
):
    target_np, footprint_np = load_or_rasterize(
        sample,
        image_size=dataset.image_size,
        taxonomy=dataset.taxonomy,
    )
    if not footprint_np.any():
        return None
    footprint = torch.from_numpy(footprint_np)[None].to(device)
    direct = predict_room_segmenter_prior(segmenter, footprint)
    features = complexity_features(footprint, None, None)
    features = features[:, : len(POLICY_INPUT_FEATURE_NAMES)]
    strength = float(policy.decide(features.to(device))["recursion_strength"][0])
    regions = propose_partition(
        footprint[0].cpu(),
        strength,
        max_depth=3,
        min_leaf_pixels=256,
    )
    local_logits = decode_partition_region_logits(
        segmenter,
        footprint,
        regions,
        global_logits=direct,
        context_fraction=0.75,
        image_size=image_size,
        recursive_blend=0.1,
    )
    graph = build_partition_graph(regions)
    node_features = region_node_features(local_logits, graph)
    dense_target = build_room_partition_target(
        target_np,
        footprint_np,
        dataset.taxonomy,
    )
    labels = []
    base_scores = []
    for region, logits in zip(regions, local_logits, strict=True):
        values = dense_target[region.mask.cpu().numpy()]
        values = values[values >= 0]
        labels.append(
            int(np.bincount(values, minlength=len(dataset.taxonomy.room_ids)).argmax())
            if values.size
            else -100
        )
        top, left, bottom, right = region.bbox
        local_mask = region.mask[top:bottom, left:right].to(device)
        selected = logits[:, local_mask]
        base_scores.append(selected.mean(dim=1))
    return {
        "name": f"{sample.relative_case}/{sample.floor_id}",
        "features": node_features.cpu(),
        "adjacency": graph.adjacency[None].cpu(),
        "topology": RecursiveSemanticGraphAdapter.topology_features(graph)[None].cpu(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "base_scores": torch.stack(base_scores).cpu(),
        "target": torch.from_numpy(dense_target),
        "footprint": torch.from_numpy(footprint_np.astype(bool)),
        "rooms": len(regions),
        "strength": strength,
    }


def move_case(case, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in case.items()
    }


def evaluate(
    adapter,
    cases,
    device,
) -> dict[str, float]:
    region_correct = 0
    region_count = 0
    base_iou = []
    refined_iou = []
    for raw_case in cases:
        case = move_case(raw_case, device)
        valid = case["labels"] >= 0
        if not valid.any():
            continue
        correction, _ = adapter.forward_features(
            case["features"],
            case["adjacency"],
            case["topology"],
            recursion_strength=case["strength"],
        )
        base_logits = case["base_scores"][valid]
        refined_logits = base_logits + correction[0, valid]
        expected = case["labels"][valid]
        region_correct += int((refined_logits.argmax(-1) == expected).sum())
        region_count += int(valid.sum())
        base_labels = base_logits.argmax(-1).cpu().numpy()
        refined_labels = refined_logits.argmax(-1).cpu().numpy()
        expected_np = expected.cpu().numpy()
        base_iou.append(room_iou(base_labels, expected_np))
        refined_iou.append(room_iou(refined_labels, expected_np))
    return {
        "region_accuracy": region_correct / max(1, region_count),
        "base_region_type_iou": float(np.mean(base_iou)) if base_iou else 0.0,
        "refined_region_type_iou": (
            float(np.mean(refined_iou)) if refined_iou else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_train,
        augment=False,
    )
    val_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit_val,
        augment=False,
    )
    segmenter, _ = load_room_segmenter_prior(
        args.checkpoint,
        train_dataset.taxonomy,
        device,
    )
    policy = load_recursion_policy(args.recursion_policy, device)
    case_cache = args.feature_cache or args.output_dir / "feature_cache.pt"
    cache_key = {
        "checkpoint": str(args.checkpoint.resolve()),
        "recursion_policy": str(args.recursion_policy.resolve()),
        "train": len(train_dataset.samples),
        "val": len(val_dataset.samples),
        "image_size": args.image_size,
    }
    if case_cache.is_file():
        cached = torch.load(case_cache, map_location="cpu", weights_only=False)
        if cached.get("key") == cache_key:
            train_cases = cached["train"]
            val_cases = cached["val"]
            print(
                json.dumps(
                    {
                        "feature_cache": str(case_cache),
                        "train_cases": len(train_cases),
                        "val_cases": len(val_cases),
                    }
                ),
                flush=True,
            )
        else:
            train_cases = None
    else:
        train_cases = None
    if train_cases is None:
        train_cases = [
            prepare_case(
                sample,
                train_dataset,
                segmenter,
                policy,
                device,
                args.image_size,
            )
            for sample in train_dataset.samples
        ]
        train_cases = [case for case in train_cases if case is not None]
        val_cases = [
            prepare_case(
                sample,
                val_dataset,
                segmenter,
                policy,
                device,
                args.image_size,
            )
            for sample in val_dataset.samples
        ]
        val_cases = [case for case in val_cases if case is not None]
        case_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "key": cache_key,
                "train": train_cases,
                "val": val_cases,
            },
            case_cache,
        )
    adapter = RecursiveSemanticGraphAdapter(
        len(train_dataset.taxonomy.room_ids),
        hidden_size=args.hidden_size,
        heads=args.heads,
        router_count=args.router_count,
        top_k=args.top_k,
        blend=args.blend,
    ).to(device)
    all_labels = torch.cat(
        [case["labels"] for case in train_cases]
    )
    all_labels = all_labels[all_labels >= 0]
    counts = torch.bincount(
        all_labels,
        minlength=len(train_dataset.taxonomy.room_ids),
    ).float()
    class_weights = (
        counts.sum() / (counts.clamp_min(1.0) * counts.numel())
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    best_metric = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        order = torch.randperm(len(train_cases)).tolist()
        losses = []
        for step, index in enumerate(order):
            case = move_case(train_cases[index], device)
            valid = case["labels"] >= 0
            if not valid.any():
                continue
            correction, _ = adapter.forward_features(
                case["features"],
                case["adjacency"],
                case["topology"],
                recursion_strength=case["strength"],
            )
            logits = (
                case["base_scores"][valid]
                + correction[0, valid]
            ).float()
            loss = F.cross_entropy(
                logits,
                case["labels"][valid],
                weight=class_weights,
            )
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(order) % args.accumulation:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(
            adapter,
            val_cases,
            device,
        )
        score = metrics["refined_region_type_iou"]
        if score > best_metric:
            best_metric = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                **metrics,
            }
        )
        print(json.dumps(history[-1]), flush=True)
    if best_state is not None:
        adapter.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": adapter.state_dict(),
        "num_classes": len(train_dataset.taxonomy.room_ids),
        "taxonomy": {
            "class_names": list(train_dataset.taxonomy.class_names),
            "room_ids": list(train_dataset.taxonomy.room_ids),
        },
        "config": {
            "hidden_size": args.hidden_size,
            "heads": args.heads,
            "router_count": args.router_count,
            "top_k": args.top_k,
            "blend": args.blend,
        },
    }
    torch.save(payload, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "open_source_adaptation": {
            "MiTA": "third_party/MiTA/mita/mita_attention_route.py",
            "implementation": "floorflow.model.official_mita_route",
        },
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "baseline": evaluate(
            RecursiveSemanticGraphAdapter(
                len(train_dataset.taxonomy.room_ids),
                hidden_size=args.hidden_size,
                blend=args.blend,
            ).to(device),
            val_cases,
            device,
        ),
        "best": evaluate(
            adapter,
            val_cases,
            device,
        ),
        "history": history,
        "selection": "train/val only; target labels are never inference inputs",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
