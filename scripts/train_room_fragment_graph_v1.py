#!/usr/bin/env python3
"""Train a predicted-room fragment merge and existence arbiter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaRasterControlDataset
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.room_fragment_graph import (
    FragmentGraph,
    FragmentGraphArbiter,
    build_fragment_graph,
    render_fragment_graph,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument(
        "--room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/train/room_fragment_graph_screen256_v1",
    )
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
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
def collect_graphs(
    room_model,
    loader,
    taxonomy,
    args,
) -> list[FragmentGraph]:
    device = torch.device(args.device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    graphs = []
    for target, footprint, image, _floor_ids, _names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        logits = room_model(footprint, control_image=image)
        probability = logits.softmax(dim=1).cpu().numpy()
        labels = room_ids[logits.argmax(dim=1)].cpu().numpy()
        footprint_np = footprint.bool().cpu().numpy()
        labels = labels * footprint_np
        target_np = target.cpu().numpy()
        for index in range(target.shape[0]):
            graph = build_fragment_graph(
                labels[index],
                probability[index],
                target_np[index],
                footprint_np[index],
                taxonomy,
                neighbors=args.neighbors,
                node_iou_threshold=args.node_iou_threshold,
                edge_iou_threshold=args.edge_iou_threshold,
            )
            if graph is not None:
                graphs.append(graph)
    return graphs


def metrics_row(
    labels: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy,
) -> dict[str, float]:
    room_ids = np.asarray(taxonomy.room_ids, dtype=np.int64)
    suite = room_metric_suite(
        labels,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(labels, room_ids),
        target_room_support=np.isin(target, room_ids),
    )
    instance = suite["matched_instance"]
    return {
        "room_iou": float(suite["visible_semantic"]["room_macro_iou"]),
        "instance_f1": float(instance["instance_f1_at_threshold"]),
        "type_accuracy": float(instance["matched_type_accuracy"]),
        "count_mae": float(instance["count_abs_error"]),
        "room_support_iou": float(
            suite["occlusion_aware_support"]["room_support_iou"]
        ),
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["room_iou"]
        + 0.75 * metrics["instance_f1"]
        + 0.1 * metrics["type_accuracy"]
        - 0.02 * metrics["count_mae"]
    )


@torch.inference_mode()
def evaluate(
    model,
    graphs: list[FragmentGraph],
    taxonomy,
    device: torch.device,
) -> dict:
    node_thresholds = (0.0, 0.3, 0.5, 0.7)
    edge_thresholds = (0.3, 0.5, 0.7)
    existence_thresholds = (0.2, 0.4, 0.6)
    arms = {"baseline": []}
    arms.update(
        {
            f"node_{node:.1f}_edge_{edge:.1f}": []
            for node in node_thresholds
            for edge in edge_thresholds
        }
    )
    if graphs and graphs[0].graph_kind == "query":
        arms.update(
            {
                f"existence_{threshold:.1f}": []
                for threshold in existence_thresholds
            }
        )
    action_rows = []
    model.eval()
    for graph in graphs:
        output = model(
            graph.node_features.to(device),
            graph.pair_features.to(device),
            graph.edge_index.to(device),
        )
        node_probability = output["node_logits"].sigmoid().cpu()
        edge_probability = output["edge_logits"].sigmoid().cpu()
        action_rows.append(
            {
                "node_probability": node_probability,
                "node_target": graph.node_target.float(),
                "edge_probability": edge_probability,
                "edge_target": graph.edge_target.float(),
            }
        )
        arms["baseline"].append(
            metrics_row(
                graph.base_labels,
                graph.target_labels,
                graph.footprint,
                taxonomy,
            )
        )
        if graph.graph_kind == "query":
            existence_logits = torch.logit(
                graph.node_existence.clamp(1.0e-6, 1.0 - 1.0e-6)
            )
            for threshold in existence_thresholds:
                labels = render_fragment_graph(
                    graph,
                    existence_logits,
                    torch.full_like(graph.edge_target.float(), -20.0),
                    taxonomy,
                    node_threshold=threshold,
                    edge_threshold=0.5,
                    class_mode="safe",
                    query_score_rendering=True,
                    fill_footprint=False,
                )
                arms[f"existence_{threshold:.1f}"].append(
                    metrics_row(
                        labels,
                        graph.target_labels,
                        graph.footprint,
                        taxonomy,
                    )
                )
        for node_threshold in node_thresholds:
            for edge_threshold in edge_thresholds:
                labels = render_fragment_graph(
                    graph,
                    output["node_logits"].cpu(),
                    output["edge_logits"].cpu(),
                    taxonomy,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                    class_mode=(
                        "safe"
                        if graph.graph_kind == "query"
                        else "guarded"
                    ),
                    query_score_rendering=graph.graph_kind == "query",
                    fill_footprint=False,
                )
                arms[f"node_{node_threshold:.1f}_edge_{edge_threshold:.1f}"].append(
                    metrics_row(
                        labels,
                        graph.target_labels,
                        graph.footprint,
                        taxonomy,
                    )
                )
    summary = {
        name: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for name, rows in arms.items()
    }
    action_summary = {}
    for threshold in node_thresholds:
        predicted = torch.cat(
            [row["node_probability"] for row in action_rows]
        ) >= threshold
        expected = torch.cat(
            [row["node_target"] for row in action_rows]
        ).bool()
        true_positive = int((predicted & expected).sum())
        action_summary[f"node_{threshold:.1f}"] = {
            "precision": true_positive / max(1, int(predicted.sum())),
            "recall": true_positive / max(1, int(expected.sum())),
        }
    nonempty_edges = [row for row in action_rows if len(row["edge_target"])]
    for threshold in edge_thresholds:
        predicted = torch.cat(
            [row["edge_probability"] for row in nonempty_edges]
        ) >= threshold
        expected = torch.cat(
            [row["edge_target"] for row in nonempty_edges]
        ).bool()
        true_positive = int((predicted & expected).sum())
        action_summary[f"edge_{threshold:.1f}"] = {
            "precision": true_positive / max(1, int(predicted.sum())),
            "recall": true_positive / max(1, int(expected.sum())),
        }
    return {"arms": summary, "actions": action_summary}


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("fragment graph training is restricted to cuda:0")
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
    room_model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    common = {
        "batch_size": 8,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    train_graphs = collect_graphs(
        room_model,
        DataLoader(train_dataset, shuffle=False, **common),
        taxonomy,
        args,
    )
    val_graphs = collect_graphs(
        room_model,
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
        seen = 0
        for start in range(0, len(train_graphs), args.batch_size):
            batch = [train_graphs[index] for index in permutation[start : start + args.batch_size]]
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
            seen += len(batch)
        validation = evaluate(model, val_graphs, taxonomy, device)
        arm_name, arm_metrics = max(
            validation["arms"].items(),
            key=lambda item: selection_score(item[1]),
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
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
        "scope": "predicted-room fragment merge and existence arbiter",
        "condition_mode": "predicted_control",
        "target_used_at_inference": False,
        "target_usage": "node and edge labels only",
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
