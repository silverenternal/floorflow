#!/usr/bin/env python3
"""Train a scorer and endpoint residual over room-boundary wall candidates."""

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
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
    load_predicted_channel_state_dict,
)
from floorflow.objectives.candidate_filtration_loss import (
    filtration_descriptor,
    graphlet_descriptor_loss,
)
from floorflow.training.cad_wall_candidate_epoch_cache import (
    cache_size,
    load_candidate_batch,
    prepare_candidate_cache,
)
from floorflow.training.cad_wall_candidate_training import (
    candidate_targets,
)

ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--predicted-root", type=Path, default=ROOT / "outputs/control_contracts/model_predicted_full_v1")
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/cache/cubicasa5k_full_v2_384")
    parser.add_argument("--geometry-root", type=Path, default=ROOT / "outputs/cache/cubicasa5k_geometry_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=256)
    parser.add_argument("--limit-val", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-walls", type=int, default=128)
    parser.add_argument("--wall-boundary-width", type=int, default=7)
    parser.add_argument("--candidate-mode", choices=("boundary", "host", "skeleton_portfolio"), default="boundary")
    parser.add_argument("--minimum-room-area", type=int, default=64)
    parser.add_argument("--room-program-bias", type=float, default=0.0)
    parser.add_argument("--use-junction-stage", action="store_true")
    parser.add_argument("--match-tolerance", type=float, default=0.08)
    parser.add_argument("--residual-scale", type=float, default=0.0)
    parser.add_argument("--score-residual-weight", type=float, default=0.01)
    parser.add_argument("--decode-mode", choices=("score", "budget", "phases", "all"), default="score")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-cache-root", type=Path, default=None)
    parser.add_argument("--use-encoded-score", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    scorer,
    graph_projection,
    layer_model,
    stage_runtime,
    loader,
    taxonomy,
    args,
    device,
    target_bank,
    optimizer=None,
    candidate_cache=None,
) -> tuple[dict[str, float], float | None]:
    scorer.train(optimizer is not None)
    totals = {}
    validation_thresholds = (
        (float(args.score_threshold),)
        if args.score_threshold is not None
        else (
            (-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0)
            if args.decode_mode in {"score", "all"}
            else ()
        )
    )
    validation_totals = {
        threshold: {}
        for threshold in validation_thresholds
    }
    validation_batches = {
        threshold: 0.0
        for threshold in validation_thresholds
    }
    budget_totals = {}
    budget_batches = 0.0
    phase_totals = {}
    phase_batches = 0.0
    control_totals = {}
    control_batches = 0.0
    seen = 0
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        candidates, tokens, opening_probability, field_features = (
            load_candidate_batch(
                cache={} if candidate_cache is None else candidate_cache,
                split="train" if optimizer is not None else "val",
                names=names,
                root=args.predicted_root,
                args=args,
                device=device,
                footprint=footprint,
                image=image,
                floor_ids=floor_ids,
                taxonomy=taxonomy,
                stage_runtime=stage_runtime,
                layer_model=layer_model,
                controls_loader=_predicted_controls,
                graph_stages=GRAPH_STAGES,
                minimum_room_area=args.minimum_room_area,
                max_walls=args.max_walls,
                disk_root=args.candidate_cache_root,
            )
        )
        outputs = scorer(
            field_features,
            candidates.endpoints,
            candidates.valid,
            candidates.evidence,
            graph_projection(tokens),
            candidates.exterior,
            proposal_source=candidates.proposal_source,
        )
        targets = target_bank.batch_targets(names, device)
        (
            expected,
            coordinate,
            target_budget,
            target_exterior_budget,
            target_interior_budget,
            target_exterior_candidates,
        ) = candidate_targets(
            outputs,
            targets,
            candidates.valid,
            args.match_tolerance,
            match_endpoints=candidates.endpoints,
        )
        valid_float = candidates.valid.float()
        positive = (expected * valid_float).sum().clamp_min(1.0)
        negative = (
            (1.0 - expected) * valid_float
        ).sum().clamp_min(1.0)
        score_elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["score_logits"],
            expected,
            pos_weight=(negative / positive).sqrt().clamp(1.0, 20.0),
            reduction="none",
        )
        score_loss = (
            score_elementwise[candidates.valid].mean()
            if bool(candidates.valid.any())
            else outputs["score_logits"].sum() * 0.0
        )
        budget_loss = torch.nn.functional.smooth_l1_loss(
            outputs["budget_fraction"],
            target_budget,
        )
        phase_budget_loss = (
            torch.nn.functional.smooth_l1_loss(
                outputs["exterior_budget_fraction"],
                target_exterior_budget,
            )
            + torch.nn.functional.smooth_l1_loss(
                outputs["interior_budget_fraction"],
                target_interior_budget,
            )
        )
        matched = expected.bool()
        wall_type_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                outputs["exterior_logits"][matched],
                target_exterior_candidates[matched],
            )
            if bool(matched.any())
            else outputs["exterior_logits"].sum() * 0.0
        )
        wall_type_accuracy = (
            (
                (
                    outputs["exterior_logits"][matched].sigmoid() >= 0.5
                )
                == (target_exterior_candidates[matched] >= 0.5)
            ).float().mean()
            if bool(matched.any())
            else outputs["exterior_logits"].sum() * 0.0
        )
        score_residual = (
            outputs["raw_score_logits"][candidates.valid].pow(2).mean()
            if bool(candidates.valid.any())
            else outputs["raw_score_logits"].sum() * 0.0
        )
        predicted_descriptor = filtration_descriptor(
            candidates.endpoints,
            outputs["score_logits"].sigmoid(),
            candidates.valid,
        )
        target_descriptor = filtration_descriptor(
            targets["wall_endpoints"],
            targets["wall_exists"].float(),
            targets["wall_exists"].bool(),
        )
        constructed_descriptor_loss = graphlet_descriptor_loss(
            predicted_descriptor,
            target_descriptor,
        )
        descriptor_loss = 0.5 * (
            constructed_descriptor_loss
            + graphlet_descriptor_loss(
                outputs["predicted_descriptor"],
                target_descriptor,
            )
        )
        absolute_descriptor_loss = torch.nn.functional.smooth_l1_loss(
            outputs["predicted_descriptor"] / float(args.max_walls),
            target_descriptor / float(args.max_walls),
        )
        loss = (
            score_loss
            + 5.0 * coordinate
            + 2.0 * budget_loss
            + 2.0 * phase_budget_loss
            + 0.5 * wall_type_loss
            + descriptor_loss
            + 0.1 * absolute_descriptor_loss
            + float(args.score_residual_weight) * score_residual
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*scorer.parameters(), *graph_projection.parameters()],
                1.0,
            )
            optimizer.step()
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * target.shape[0]
        totals["score_loss"] = totals.get("score_loss", 0.0) + float(score_loss.detach()) * target.shape[0]
        totals["coordinate_loss"] = totals.get("coordinate_loss", 0.0) + float(coordinate.detach()) * target.shape[0]
        totals["budget_loss"] = totals.get("budget_loss", 0.0) + float(budget_loss.detach()) * target.shape[0]
        totals["phase_budget_loss"] = totals.get("phase_budget_loss", 0.0) + float(phase_budget_loss.detach()) * target.shape[0]
        totals["wall_type_loss"] = totals.get("wall_type_loss", 0.0) + float(wall_type_loss.detach()) * target.shape[0]
        totals["score_residual"] = totals.get("score_residual", 0.0) + float(score_residual.detach()) * target.shape[0]
        totals["wall_type_accuracy"] = totals.get("wall_type_accuracy", 0.0) + float(wall_type_accuracy.detach()) * target.shape[0]
        totals["budget_mae"] = totals.get("budget_mae", 0.0) + float(
            (
                outputs["budget_count"]
                - target_budget * float(args.max_walls)
            ).abs().mean().detach()
        ) * target.shape[0]
        totals["phase_budget_mae"] = totals.get("phase_budget_mae", 0.0) + float(
            (
                outputs["exterior_budget_count"]
                - target_exterior_budget * float(args.max_walls)
            ).abs().mean().detach()
            + (
                outputs["interior_budget_count"]
                - target_interior_budget * float(args.max_walls)
            ).abs().mean().detach()
        ) * target.shape[0]
        totals["descriptor_mae"] = totals.get("descriptor_mae", 0.0) + float(
            (
                outputs["predicted_descriptor"] - target_descriptor
            ).abs().mean().detach()
        ) * target.shape[0]
        totals["descriptor_loss"] = totals.get("descriptor_loss", 0.0) + float(descriptor_loss.detach()) * target.shape[0]
        totals["absolute_descriptor_loss"] = totals.get("absolute_descriptor_loss", 0.0) + float(absolute_descriptor_loss.detach()) * target.shape[0]
        if optimizer is None:
            batch_size = float(target.shape[0])
            for threshold in validation_thresholds:
                batch_metrics = candidate_plan_metrics(
                    outputs["endpoints"],
                    outputs["score_logits"],
                    candidates.valid,
                    candidates.rooms,
                    candidates.host_walls,
                    opening_probability,
                    target,
                    taxonomy,
                    threshold,
                    wall_type_logits=outputs["exterior_logits"],
                )
                for key, value in batch_metrics.items():
                    validation_totals[threshold][key] = (
                        validation_totals[threshold].get(key, 0.0)
                        + float(value) * batch_size
                    )
                validation_batches[threshold] += batch_size
            budget_metrics = (
                candidate_plan_metrics(
                    outputs["endpoints"],
                    outputs["score_logits"],
                    candidates.valid,
                    candidates.rooms,
                    candidates.host_walls,
                    opening_probability,
                    target,
                    taxonomy,
                    0.0,
                    wall_budget=(
                        outputs["budget_count"].detach().cpu().round().long()
                    ),
                    wall_type_logits=outputs["exterior_logits"],
                )
                if args.decode_mode in {"budget", "all"}
                else {}
            )
            for key, value in budget_metrics.items():
                budget_totals[key] = (
                    budget_totals.get(key, 0.0)
                    + float(value) * batch_size
                )
            budget_batches += batch_size
            phase_metrics = (
                candidate_plan_metrics(
                    outputs["endpoints"],
                    outputs["score_logits"],
                    candidates.valid,
                    candidates.rooms,
                    candidates.host_walls,
                    opening_probability,
                    target,
                    taxonomy,
                    0.0,
                    exterior=candidates.exterior,
                    exterior_budget=(
                        outputs["exterior_budget_count"]
                        .detach()
                        .cpu()
                        .round()
                        .long()
                    ),
                    interior_budget=(
                        outputs["interior_budget_count"]
                        .detach()
                        .cpu()
                        .round()
                        .long()
                    ),
                    wall_type_logits=outputs["exterior_logits"],
                )
                if args.decode_mode in {"phases", "all"}
                else {}
            )
            for key, value in phase_metrics.items():
                phase_totals[key] = (
                    phase_totals.get(key, 0.0)
                    + float(value) * batch_size
                )
            phase_batches += batch_size
            control_metrics = candidate_plan_metrics(
                candidates.control_endpoints,
                candidates.control_evidence,
                candidates.control_valid,
                candidates.rooms,
                candidates.host_walls,
                opening_probability,
                target,
                taxonomy,
                0.2,
                wall_type_logits=None,
            )
            for key, value in control_metrics.items():
                control_totals[key] = (
                    control_totals.get(key, 0.0)
                    + float(value) * batch_size
                )
            control_batches += batch_size
        seen += target.shape[0]
    metrics = {key: value / max(1, seen) for key, value in totals.items()}
    if optimizer is not None:
        return metrics, None
    best = None
    for threshold in validation_thresholds:
        metrics_for_threshold = {
            key: value / max(1.0, validation_batches[threshold])
            for key, value in validation_totals[threshold].items()
        }
        score = (
            metrics_for_threshold["eligible_wall"]
            + 0.1 * metrics_for_threshold["eligible_opening"]
        )
        if best is None or score > best[0]:
            best = (score, f"threshold={threshold}", metrics_for_threshold)
    budget_metrics = {
        key: value / max(1.0, budget_batches)
        for key, value in budget_totals.items()
    }
    if budget_metrics:
        budget_score = (
            budget_metrics["eligible_wall"]
            + 0.1 * budget_metrics["eligible_opening"]
        )
        if best is None or budget_score > best[0]:
            best = (budget_score, "budget", budget_metrics)
    phase_metrics = {
        key: value / max(1.0, phase_batches)
        for key, value in phase_totals.items()
    }
    if phase_metrics:
        phase_score = (
            phase_metrics["eligible_wall"]
            + 0.1 * phase_metrics["eligible_opening"]
        )
        if best is None or phase_score > best[0]:
            best = (phase_score, "phases", phase_metrics)
    assert best is not None
    control_metrics = {
        f"control_{key}": value / max(1.0, control_batches)
        for key, value in control_totals.items()
    }
    selection_mode = best[1]
    return {
        **metrics,
        **best[2],
        **control_metrics,
        **{
            f"budget_{key}": value
            for key, value in budget_metrics.items()
        },
        **{
            f"phase_{key}": value
            for key, value in phase_metrics.items()
        },
        "selection_mode": selection_mode,
        "predicted_wall_budget_mean": float(
            outputs["budget_count"].mean().detach().cpu()
        ),
    }, best[0]


def main() -> None:
    args = parse_args()
    if args.candidate_cache_root is not None:
        prepare_candidate_cache(args.candidate_cache_root, args, args.layer_checkpoint)
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.layer_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    load_predicted_channel_state_dict(layer_model, payload["model"])
    layer_model.requires_grad_(False).eval()
    stage_runtime = CadStageRuntime(
        taxonomy,
        device,
        room_checkpoint=config["room_checkpoint"],
        structure_checkpoint=config["structure_checkpoint"],
        centerline_checkpoint=config["centerline_checkpoint"],
        opening_checkpoint=config["opening_checkpoint"],
        room_blend=float(config["room_blend"]),
        structure_size=128,
        room_program_bias=args.room_program_bias,
        use_junction_stage=args.use_junction_stage,
    )
    scorer = CadWallCandidateScorer(
        int(config["field_channels"]),
        hidden_dim=args.hidden_dim,
        graph_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        use_encoded_score=args.use_encoded_score,
        source_features=args.candidate_mode == "skeleton_portfolio",
    ).to(device)
    graph_projection = torch.nn.Linear(
        int(config["field_channels"]),
        args.hidden_dim,
    ).to(device)
    train = RasterPredictedControlDataset(
        args.predicted_root,
        "train",
        args.cache_root,
        args.data_root,
        image_size=int(config["image_size"]),
        limit=args.limit_train,
    )
    val = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        image_size=int(config["image_size"]),
        limit=args.limit_val,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
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
        [*scorer.parameters(), *graph_projection.parameters()],
        lr=args.lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    best_epoch = -1
    train_cache = {}
    val_cache = {}
    for epoch in range(args.epochs):
        train_metrics, _ = run_epoch(
            scorer, graph_projection, layer_model, stage_runtime,
            DataLoader(train, shuffle=True, **loader_args),
            taxonomy, args, device, train_bank, optimizer,
            candidate_cache=train_cache,
        )
        val_metrics, score = run_epoch(
            scorer, graph_projection, layer_model, stage_runtime,
            DataLoader(val, shuffle=False, **loader_args),
            taxonomy, args, device, val_bank,
            candidate_cache=val_cache,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "scorer": scorer.state_dict(),
            "graph_projection": graph_projection.state_dict(),
            "config": {**vars(args), "field_channels": int(config["field_channels"])},
            "selection_mode": val_metrics["selection_mode"],
            "metrics": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score is not None and score > best:
            best = score
            best_epoch = epoch
            torch.save(checkpoint, args.output_dir / "best.pt")
        if best_epoch >= 0 and epoch - best_epoch >= args.early_stop_patience:
            break
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "test_opened": False,
                "target_used_at_inference": False,
                "history": history,
                "best_score": best,
                "best_epoch": best_epoch,
                "early_stopped": len(history) < args.epochs,
                "candidate_cache_entries": {
                    "train": cache_size(train_cache),
                    "val": cache_size(val_cache),
                },
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
