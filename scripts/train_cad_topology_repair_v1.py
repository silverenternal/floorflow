#!/usr/bin/env python3
"""Train and evaluate bounded CAD topology repair on candidate wall graphs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.data.vector_primitive_target_bank import VectorPrimitiveTargetBank
from floorflow.evaluation.cad_candidate_plan import candidate_plan_metrics
from floorflow.heads.cad_topology_repair import (
    CadTopologyRepairHead,
    apply_bounded_topology_repair,
)
from floorflow.inference.cad_stage_runtime import CadStageRuntime
from floorflow.model.predicted_channel_checkpoint import (
    build_predicted_channel_model,
)
from floorflow.objectives.candidate_filtration_loss import endpoint_adjacency
from floorflow.training.cad_wall_candidate_training import (
    build_candidate_batch,
    match_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".data/cubicasa5k",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2_384",
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_geometry_v1",
    )
    parser.add_argument("--limit-train", type=int, default=64)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-walls", type=int, default=128)
    parser.add_argument("--minimum-room-area", type=int, default=64)
    parser.add_argument("--match-tolerance", type=float, default=0.08)
    parser.add_argument("--coordinate-weight", type=float, default=1.0)
    parser.add_argument("--unmatched-delta-weight", type=float, default=0.01)
    parser.add_argument("--max-edits", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2905)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_inputs(
    layer_output: dict,
    candidates,
) -> dict[str, torch.Tensor]:
    endpoints = candidates.endpoints
    valid = candidates.valid
    evidence = candidates.evidence
    delta = endpoints[:, :, 1] - endpoints[:, :, 0]
    length = delta.norm(dim=-1)
    midpoint = 0.5 * (endpoints[:, :, 0] + endpoints[:, :, 1])
    adjacency = endpoint_adjacency(endpoints, valid)
    degree = adjacency.sum(dim=-1) / 4.0
    features = torch.stack(
        [
            evidence,
            length,
            delta[..., 0],
            delta[..., 1],
            midpoint[..., 0],
            midpoint[..., 1],
            candidates.exterior.float(),
            degree,
        ],
        dim=-1,
    )
    host_prior = evidence.masked_fill(~valid, 0.0).clamp_min(1.0e-4)
    host_prior = host_prior / host_prior.sum(dim=1, keepdim=True).clamp_min(1.0e-4)
    diagnostics = torch.stack(
        [
            evidence.mean(dim=1),
            valid.float().mean(dim=1),
            length.mean(dim=1),
            degree.mean(dim=1),
            candidates.exterior.float().mean(dim=1),
            host_prior.max(dim=1).values,
        ],
        dim=-1,
    )
    return {
        "candidate_features": features,
        "endpoints": endpoints,
        "valid": valid,
        "wall_probability": evidence,
        "junction_degree": degree,
        "opening_attachment": host_prior[:, None],
        "validity_diagnostics": diagnostics,
        "validation_token": layer_output["validation_token"],
    }


def repair_targets(candidates, targets, tolerance):
    pairs, positive_rows = match_candidates(
        candidates.endpoints,
        targets["wall_endpoints"],
        targets["wall_exists"],
        candidates.valid,
        tolerance,
    )
    positive = torch.from_numpy(np.stack(positive_rows)).to(candidates.endpoints)
    delta_target = torch.zeros_like(candidates.endpoints)
    matched = torch.zeros_like(candidates.valid)
    for batch, (rows, columns) in enumerate(pairs):
        for row, column in zip(rows, columns, strict=True):
            predicted = candidates.endpoints[batch, row]
            target = targets["wall_endpoints"][batch, column]
            direct = (predicted - target).abs().mean()
            reverse = (predicted - target[[1, 0]]).abs().mean()
            oriented = target if direct <= reverse else target[[1, 0]]
            delta_target[batch, row] = oriented - predicted
            matched[batch, row] = True
    return positive.float(), delta_target, matched


def train_metrics(
    head,
    inputs,
    positive,
    delta_target,
    matched,
    args,
):
    output = head(**inputs)
    valid = inputs["valid"]
    positive_count = (positive * valid).sum().clamp_min(1.0)
    negative_count = ((1.0 - positive) * valid).sum().clamp_min(1.0)
    keep_loss = F.binary_cross_entropy_with_logits(
        output["wall_keep_logits"],
        positive,
        pos_weight=(negative_count / positive_count).sqrt().clamp(1.0, 20.0),
    )
    coordinate_loss = (
        F.smooth_l1_loss(
            output["endpoint_delta"][matched],
            delta_target[matched],
        )
        if bool(matched.any())
        else output["endpoint_delta"].sum() * 0.0
    )
    unmatched = ~matched
    unmatched_loss = (
        output["endpoint_delta"][unmatched].square().mean()
        if bool(unmatched.any())
        else output["endpoint_delta"].sum() * 0.0
    )
    loss = (
        keep_loss
        + float(args.coordinate_weight) * coordinate_loss
        + float(args.unmatched_delta_weight) * unmatched_loss
    )
    return loss, output, {
        "keep_loss": keep_loss.detach(),
        "coordinate_loss": coordinate_loss.detach(),
        "unmatched_delta_loss": unmatched_loss.detach(),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


@torch.inference_mode()
def evaluate(
    head,
    layer_model,
    stage_runtime,
    loader,
    taxonomy,
    target_bank,
    args,
    device,
):
    head.eval()
    control_rows = []
    repair_rows = []
    edit_rows = []
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        controls = _predicted_controls(
            args.predicted_root,
            "val",
            names,
            args,
            device,
        )
        controls, _ = stage_runtime(
            controls,
            footprint,
            image.to(device),
            floor_ids.to(device),
        )
        layer_output = layer_model(controls, footprint)
        candidates = build_candidate_batch(
            controls,
            layer_output,
            footprint,
            taxonomy,
            minimum_room_area=args.minimum_room_area,
            max_walls=args.max_walls,
            device=device,
        )
        inputs = candidate_inputs(layer_output, candidates)
        repair = head(**inputs)
        opening = F.interpolate(
            controls["opening_family_probability"].float(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        common = {
            "endpoints": candidates.endpoints,
            "scores": candidates.evidence,
            "valid": candidates.valid,
            "rooms": candidates.rooms,
            "host_walls": candidates.host_walls,
            "opening_probability": opening,
            "target": target,
            "taxonomy": taxonomy,
            "threshold": 0.2,
            "repair_max_edits": args.max_edits,
        }
        control_rows.append(candidate_plan_metrics(**common))
        repair_rows.append(
            candidate_plan_metrics(
                **common,
                repair_keep_logits=repair["wall_keep_logits"],
                repair_endpoint_delta=repair["endpoint_delta"],
            )
        )
        bounded = apply_bounded_topology_repair(
            candidates.endpoints,
            candidates.valid,
            repair["wall_keep_logits"],
            repair["endpoint_delta"],
            max_edits=args.max_edits,
        )
        edit_rows.append(
            {
                "edit_rate": float(
                    bounded["edited"].sum()
                    / candidates.valid.sum().clamp_min(1)
                ),
                "deleted_walls": float(bounded["removed"].sum()),
                "moved_endpoints": float(bounded["moved"].sum()),
                "max_edits_per_case": float(
                    bounded["edited"].sum(dim=1).max()
                ),
            }
        )
    control = aggregate(control_rows)
    repaired = aggregate(repair_rows)
    edits = aggregate(edit_rows)
    return {
        "control": control,
        "repair": repaired,
        "repair_minus_control": {
            key: repaired[key] - control[key]
            for key in control
        },
        "edits": edits,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.layer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = payload["config"]
    taxonomy = FloorplanTaxonomy.from_json(payload["taxonomy"])
    layer_model = build_predicted_channel_model(taxonomy, config).to(device)
    layer_model.load_state_dict(payload["model"])
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
    )
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
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train, shuffle=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    train_bank = VectorPrimitiveTargetBank(
        args.geometry_root,
        "train",
        image_size=int(config["image_size"]),
        max_walls=args.max_walls,
    )
    val_bank = VectorPrimitiveTargetBank(
        args.geometry_root,
        "val",
        image_size=int(config["image_size"]),
        max_walls=args.max_walls,
    )
    head = CadTopologyRepairHead(
        8,
        int(config["field_channels"]),
        hidden_dim=args.hidden_dim,
        heads=4,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        started = time.perf_counter()
        head.train()
        totals = {}
        seen = 0
        for target, footprint, floor_ids, names, image in train_loader:
            footprint = footprint.to(device)
            controls = _predicted_controls(
                args.predicted_root,
                "train",
                names,
                args,
                device,
            )
            controls, _ = stage_runtime(
                controls,
                footprint,
                image.to(device),
                floor_ids.to(device),
            )
            with torch.no_grad():
                layer_output = layer_model(controls, footprint)
                candidates = build_candidate_batch(
                    controls,
                    layer_output,
                    footprint,
                    taxonomy,
                    minimum_room_area=args.minimum_room_area,
                    max_walls=args.max_walls,
                    device=device,
                )
            inputs = candidate_inputs(layer_output, candidates)
            targets = train_bank.batch_targets(names, device)
            positive, delta_target, matched = repair_targets(
                candidates,
                targets,
                args.match_tolerance,
            )
            loss, _output, components = train_metrics(
                head,
                inputs,
                positive,
                delta_target,
                matched,
                args,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            batch = footprint.shape[0]
            seen += batch
            totals["loss"] = (
                totals.get("loss", 0.0)
                + float(loss.detach())
                * batch
            )
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch
        record = {
            "epoch": epoch,
            **{key: value / max(1, seen) for key, value in totals.items()},
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "head": head.state_dict(),
            "config": vars(args),
            "taxonomy": taxonomy.to_json(),
            "epoch": epoch,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    best = torch.load(
        args.output_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    head.load_state_dict(best["head"])
    evaluation = evaluate(
        head,
        layer_model,
        stage_runtime,
        val_loader,
        taxonomy,
        val_bank,
        args,
        device,
    )
    report = {
        "schema_version": "cad-topology-repair-v1",
        "task": "PC-06M",
        "test_opened": False,
        "source_episode": str(args.layer_checkpoint),
        "control_mode": "deterministic wall-candidate evidence threshold 0.2",
        "max_edits": args.max_edits,
        "history": history,
        "selected_epoch": int(best["epoch"]),
        "evaluation": evaluation,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
