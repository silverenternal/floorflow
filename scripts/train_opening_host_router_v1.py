#!/usr/bin/env python3
"""Train many-query opening tokens to route onto final candidate walls."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls
from train_vector_topology_scaffold_v1 import GRAPH_STAGES

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.data.vector_primitive_target_bank import VectorPrimitiveTargetBank
from floorflow.evaluation.cad_candidate_plan import candidate_plan_metrics
from floorflow.heads.cad_wall_candidate_scorer import CadWallCandidateScorer
from floorflow.heads.opening_host_router import (
    OpeningHostRouter,
    opening_host_router_loss,
    opening_route_metrics,
)
from floorflow.heads.opening_query_encoder import OpeningQueryEncoder
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.inference.opening_host_projection import (
    project_routed_openings_to_final_walls,
)
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.training.cad_wall_candidate_training import build_candidate_batch
from floorflow.training.opening_host_training import (
    candidate_host_walls,
    detection_opening_queries,
    match_opening_host_targets,
    target_opening_host_targets,
    target_opening_queries,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, default=ROOT / "outputs/cache/cubicasa5k_geometry_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--route-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-router-count", type=int, default=16)
    parser.add_argument("--route-top-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(args, device):
    layer_payload = torch.load(args.layer_checkpoint, map_location="cpu", weights_only=False)
    scorer_payload = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    config = layer_payload["config"]
    scorer_config = scorer_payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(layer_payload["taxonomy"])
    layer = build_predicted_channel_model(taxonomy, config).to(device)
    layer.load_state_dict(layer_payload["model"])
    layer.requires_grad_(False).eval()
    scorer = CadWallCandidateScorer(
        int(config["field_channels"]),
        hidden_dim=int(scorer_config["hidden_dim"]),
        graph_dim=int(scorer_config["hidden_dim"]),
        residual_scale=float(scorer_config.get("residual_scale", 0.0)),
    ).to(device)
    scorer.load_state_dict(scorer_payload["scorer"])
    scorer.requires_grad_(False).eval()
    graph_projection = torch.nn.Linear(
        int(config["field_channels"]),
        int(scorer_config["hidden_dim"]),
    ).to(device)
    graph_projection.load_state_dict(scorer_payload["graph_projection"])
    graph_projection.requires_grad_(False).eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
    )
    return layer, scorer, graph_projection, stage_runtime, config, scorer_config, taxonomy


def _threshold(payload) -> float:
    mode = str(payload.get("selection_mode", "threshold=-1.0"))
    return float(mode.split("=", 1)[1]) if mode.startswith("threshold=") else -1.0


def _forward_batch(
    batch,
    layer,
    scorer,
    graph_projection,
    stage_runtime,
    config,
    scorer_config,
    taxonomy,
    query_encoder,
    router,
    device,
):
    target, footprint, floor_ids, names, image = batch
    target = target.to(device)
    footprint = footprint.to(device)
    controls = _predicted_controls(
        Path(config["predicted_root"]),
        "train" if query_encoder.training else "val",
        names,
        type(
            "Args",
            (),
            {
                "predicted_root": Path(config["predicted_root"]),
                "room_size": 224,
                "structure_size": 128,
            },
        )(),
        device,
    )
    controls, _ = stage_runtime(
        controls,
        footprint,
        image.to(device),
        floor_ids.to(device),
    )
    with torch.no_grad():
        layer_output = layer(controls, footprint)
        tokens = torch.stack(
            [layer_output["action_tokens"][name] for name in GRAPH_STAGES],
            dim=1,
        )
    candidates = build_candidate_batch(
        controls,
        layer_output,
        footprint,
        taxonomy,
        minimum_room_area=int(scorer_config["minimum_room_area"]),
        max_walls=int(scorer_config["max_walls"]),
        device=device,
        candidate_mode=scorer_config.get("candidate_mode", "boundary"),
    )
    with torch.no_grad():
        scored = scorer(
            layer_output["field_features"],
            candidates.endpoints,
            candidates.valid,
            candidates.evidence,
            graph_projection(tokens),
            candidates.exterior,
        )
    selected = candidates.valid & (scored["score_logits"] >= _threshold(scorer_payload_global))
    final_host_walls = candidate_host_walls(candidates, selected, frame=float(target.shape[-1]))
    queries = detection_opening_queries(
        controls,
        candidates,
        taxonomy,
        frame=float(target.shape[-1]),
        max_openings=48,
        device=device,
        host_walls=final_host_walls,
    )
    opening_tokens = query_encoder(
        layer_output["field_features"],
        queries.centers,
        queries.widths,
        queries.family_logits,
        queries.valid,
    )
    routed = router(
        scored["candidate_tokens"],
        opening_tokens,
        candidates.endpoints,
        opening_centers=queries.centers,
        opening_widths=queries.widths,
        wall_mask=selected,
        opening_mask=queries.valid,
    )
    return (
        target,
        controls,
        candidates,
        scored,
        queries,
        routed,
        selected,
        layer_output["field_features"],
    )


def _loss(routed, candidates, queries, selected, targets, taxonomy):
    host_targets = match_opening_host_targets(
        queries,
        candidates,
        targets,
        taxonomy,
        candidate_mask=selected,
    )
    loss, components = opening_host_router_loss(
        routed,
        host_targets.hosts.to(routed["host_logits"].device),
        host_targets.center_fraction.to(routed["host_logits"].device),
        host_targets.width_fraction.to(routed["host_logits"].device),
        host_targets.types.to(routed["host_logits"].device),
        host_targets.matched.to(routed["host_logits"].device),
    )
    return loss, components, host_targets


def _routed_rows(
    routed,
    selected,
    candidates,
    taxonomy,
    frame,
    queries,
):
    return project_routed_openings_to_final_walls(
        routed,
        candidates.endpoints,
        selected,
        taxonomy,
        frame=frame,
        query_centers=queries.centers,
        query_widths=queries.widths,
        query_families=queries.family_logits.argmax(dim=-1),
    )


def run_epoch(models, query_encoder, router, loader, target_bank, device, optimizer=None):
    query_encoder.train(optimizer is not None)
    router.train(optimizer is not None)
    totals = {}
    routed_totals = {}
    fallback_totals = {}
    plan_seen = 0
    matched_count = 0
    host_correct = 0
    seen = 0
    for batch in loader:
        (
            target,
            controls,
            candidates,
            scored,
            queries,
            routed,
            selected,
            features,
        ) = _forward_batch(batch, *models, query_encoder, router, device)
        for key, value in opening_route_metrics(routed).items():
            totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
        targets = target_bank.batch_targets(batch[3], device)
        loss, components, host_targets = _loss(
            routed,
            candidates,
            queries,
            selected,
            targets,
            models[-1],
        )
        target_queries = target_opening_queries(
            targets,
            models[-1],
            device=device,
        )
        target_tokens = query_encoder(
            features,
            target_queries.centers,
            target_queries.widths,
            target_queries.family_logits,
            target_queries.valid,
        )
        teacher_routed = router(
            scored["candidate_tokens"],
            target_tokens,
            candidates.endpoints,
            opening_centers=target_queries.centers,
            opening_widths=target_queries.widths,
            wall_mask=selected,
            opening_mask=target_queries.valid,
        )
        teacher_targets = target_opening_host_targets(
            candidates,
            targets,
            candidate_mask=selected,
        )
        teacher_loss, teacher_components = opening_host_router_loss(
            teacher_routed,
            teacher_targets.hosts.to(device),
            teacher_targets.center_fraction.to(device),
            teacher_targets.width_fraction.to(device),
            teacher_targets.types.to(device),
            teacher_targets.matched.to(device),
        )
        loss = loss + 0.5 * teacher_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*query_encoder.parameters(), *router.parameters()],
                1.0,
            )
            optimizer.step()
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * target.shape[0]
        totals["teacher_loss"] = totals.get("teacher_loss", 0.0) + float(
            teacher_loss.detach()
        ) * target.shape[0]
        totals["query_count"] = totals.get("query_count", 0.0) + float(
            queries.valid.sum()
        )
        matched = host_targets.matched.to(device)
        matched_count += int(matched.sum())
        host_correct += int(
            (
                routed["host_index"][matched]
                == host_targets.hosts.to(device)[matched]
            ).sum()
        )
        teacher_matched = teacher_targets.matched.to(device)
        teacher_correct = int(
            (
                teacher_routed["host_index"][teacher_matched]
                == teacher_targets.hosts.to(device)[teacher_matched]
            ).sum()
        )
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value) * target.shape[0]
        for key, value in teacher_components.items():
            totals[f"teacher_{key}"] = (
                totals.get(f"teacher_{key}", 0.0)
                + float(value) * target.shape[0]
            )
        seen += target.shape[0]
        totals["teacher_host_correct"] = (
            totals.get("teacher_host_correct", 0.0) + teacher_correct
        )
        totals["teacher_host_total"] = (
            totals.get("teacher_host_total", 0.0)
            + int(teacher_matched.sum())
        )
        if optimizer is None:
            opening_probability = torch.nn.functional.interpolate(
                controls["opening_family_probability"].float(),
                size=tuple(target.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
            rows = _routed_rows(
                routed,
                selected,
                candidates,
                models[-1],
                float(target.shape[-1]),
                queries,
            )
            routed_metrics = candidate_plan_metrics(
                candidates.endpoints,
                scored["score_logits"],
                candidates.valid,
                candidates.rooms,
                candidates.host_walls,
                opening_probability,
                target,
                models[-1],
                _threshold(scorer_payload_global),
                wall_type_logits=scored["exterior_logits"],
                routed_openings=rows,
            )
            fallback_metrics = candidate_plan_metrics(
                candidates.endpoints,
                scored["score_logits"],
                candidates.valid,
                candidates.rooms,
                candidates.host_walls,
                opening_probability,
                target,
                models[-1],
                _threshold(scorer_payload_global),
                wall_type_logits=scored["exterior_logits"],
            )
            batch_size = float(target.shape[0])
            for key, value in routed_metrics.items():
                routed_totals[key] = (
                    routed_totals.get(key, 0.0)
                    + float(value) * batch_size
                )
                fallback_totals[key] = (
                    fallback_totals.get(key, 0.0)
                    + float(fallback_metrics[key]) * batch_size
                )
            plan_seen += target.shape[0]
    teacher_host_accuracy = totals.pop(
        "teacher_host_correct",
        0.0,
    ) / max(1.0, totals.pop("teacher_host_total", 0.0))
    metrics = {key: value / max(1, seen) for key, value in totals.items()}
    metrics["host_accuracy"] = host_correct / max(1, matched_count)
    metrics["teacher_host_accuracy"] = teacher_host_accuracy
    if optimizer is not None:
        return metrics
    routed_metrics = {
        f"routed_{key}": value / max(1, plan_seen)
        for key, value in routed_totals.items()
    }
    fallback_metrics = {
        f"fallback_{key}": value / max(1, plan_seen)
        for key, value in fallback_totals.items()
    }
    return {**metrics, **routed_metrics, **fallback_metrics}


scorer_payload_global = {}


def main() -> None:
    global scorer_payload_global
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    scorer_payload_global = torch.load(
        args.scorer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    layer, scorer, graph_projection, stage_runtime, config, scorer_config, taxonomy = load_models(args, device)
    query_encoder = OpeningQueryEncoder(
        int(config["field_channels"]),
        hidden_dim=args.hidden_dim,
    ).to(device)
    router = OpeningHostRouter(
        int(scorer_config["hidden_dim"]),
        num_types=max(1, len(taxonomy.opening_ids)),
        route_attention=args.route_attention,
        route_router_count=args.route_router_count,
        route_top_k=args.route_top_k,
    ).to(device)
    models = (layer, scorer, graph_projection, stage_runtime, config, scorer_config, taxonomy)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "train",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit_train,
    )
    val = RasterPredictedControlDataset(
        Path(config["predicted_root"]),
        "val",
        Path(config["cache_root"]),
        Path(config["data_root"]),
        image_size=int(config["image_size"]),
        limit=args.limit_val,
    )
    train_bank = VectorPrimitiveTargetBank(
        args.geometry_root,
        "train",
        image_size=int(config["image_size"]),
    )
    val_bank = VectorPrimitiveTargetBank(
        args.geometry_root,
        "val",
        image_size=int(config["image_size"]),
    )
    optimizer = torch.optim.AdamW(
        [*query_encoder.parameters(), *router.parameters()],
        lr=args.lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            models,
            query_encoder,
            router,
            DataLoader(train, shuffle=True, **common),
            train_bank,
            device,
            optimizer,
        )
        val_metrics = run_epoch(
            models,
            query_encoder,
            router,
            DataLoader(val, shuffle=False, **common),
            val_bank,
            device,
        )
        score = float(
            val_metrics["routed_eligible_opening"]
            + 0.2 * val_metrics["routed_eligible_wall"]
            + 0.1 * val_metrics["routed_eligible_valid"]
            + 0.1 * val_metrics["routed_eligible_watertight"]
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "query_encoder": query_encoder.state_dict(),
            "router": router.state_dict(),
            "config": {
                **vars(args),
                "field_channels": int(config["field_channels"]),
                "wall_hidden_dim": int(scorer_config["hidden_dim"]),
            },
            "metrics": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best:
            best = score
            torch.save(checkpoint, args.output_dir / "best.pt")
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "opening-host-router-screen-v1",
                "test_opened": False,
                "target_used_at_inference": False,
                "best_score": best,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
