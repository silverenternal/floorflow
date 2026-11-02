#!/usr/bin/env python3
"""Train existence and duplicate-merge arbitration over query room masks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_room_fragment_graph_v1 import evaluate, selection_score

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.heads.query_mask_graph import build_query_mask_graph
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
)
from floorflow.heads.room_fragment_graph import (
    FragmentGraph,
    FragmentGraphArbiter,
)

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
        "--query-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/room_query_fullres_softroute_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/query_mask_graph_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--node-iou-threshold", type=float, default=0.2)
    parser.add_argument("--edge-iou-threshold", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--node-positive-weight", type=float, default=1.0)
    parser.add_argument("--edge-positive-weight", type=float, default=5.0)
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def collect_query_graphs(
    query_model,
    loader,
    taxonomy,
    args,
) -> list[FragmentGraph]:
    device = torch.device(args.device)
    graphs = []
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        output = query_model(footprint, control_image=image)
        footprint_np = footprint.bool().cpu().numpy()
        target_np = target.cpu().numpy()
        for index in range(target.shape[0]):
            graph = build_query_mask_graph(
                output["mask_logits"][index],
                output["class_logits"][index],
                output["query_existence_logits"][index],
                target_np[index],
                footprint_np[index],
                taxonomy,
                mask_threshold=args.mask_threshold,
                neighbors=args.neighbors,
                node_iou_threshold=args.node_iou_threshold,
                edge_iou_threshold=args.edge_iou_threshold,
            )
            if graph is not None:
                graphs.append(graph)
    return graphs


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("query mask graph training is restricted to cuda:0")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_train,
        image_mode="native",
    )
    taxonomy = train_dataset.taxonomy
    val_dataset = CubiCasaRasterControlDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        cache_root=args.cache_root,
        augment=False,
        limit=args.limit_val,
        image_mode="native",
    )
    query_model = load_raster_instance_control_model(
        args.query_checkpoint,
        taxonomy,
        device,
    )
    common = {
        "batch_size": 8,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    train_graphs = collect_query_graphs(
        query_model,
        DataLoader(train_dataset, shuffle=False, **common),
        taxonomy,
        args,
    )
    val_graphs = collect_query_graphs(
        query_model,
        DataLoader(val_dataset, shuffle=False, **common),
        taxonomy,
        args,
    )
    model = FragmentGraphArbiter(
        train_graphs[0].node_features.shape[1],
        train_graphs[0].pair_features.shape[1],
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.default_rng(epoch).permutation(len(train_graphs))
        total_loss = 0.0
        for start in range(0, len(train_graphs), args.batch_size):
            batch = [
                train_graphs[index]
                for index in permutation[start : start + args.batch_size]
            ]
            losses = []
            for graph in batch:
                output = model(
                    graph.node_features.to(device),
                    graph.pair_features.to(device),
                    graph.edge_index.to(device),
                )
                node_loss = F.binary_cross_entropy_with_logits(
                    output["node_logits"],
                    graph.node_target.float().to(device),
                    pos_weight=torch.tensor(
                        args.node_positive_weight,
                        device=device,
                    ),
                )
                edge_loss = output["edge_logits"].sum() * 0.0
                if len(graph.edge_target):
                    edge_loss = F.binary_cross_entropy_with_logits(
                        output["edge_logits"],
                        graph.edge_target.float().to(device),
                        pos_weight=torch.tensor(
                            args.edge_positive_weight,
                            device=device,
                        ),
                    )
                losses.append(node_loss + args.edge_loss_weight * edge_loss)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        validation = evaluate(model, val_graphs, taxonomy, device)
        arm_name, arm_metrics = max(
            validation["arms"].items(),
            key=lambda item: selection_score(item[1]),
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_graphs)),
            "best_arm": arm_name,
            "selection_score": selection_score(arm_metrics),
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "metrics": validation,
        }
        torch.save(state, args.output_dir / "last.pt")
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            torch.save(state, args.output_dir / "best.pt")
    report = {
        "status": "complete",
        "scope": "query-mask instance graph and calibrated existence",
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
        "target_usage": "query keep and duplicate-merge labels only",
        "train_graphs": len(train_graphs),
        "validation_graphs": len(val_graphs),
        "best_selection_score": best_score,
        "history": history,
        "config": vars(args),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
