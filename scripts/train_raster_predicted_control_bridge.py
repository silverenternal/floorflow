#!/usr/bin/env python3
"""Train the existing joint control refiner on native-raster room controls."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_residential_predicted_refiner import (
    _binary_f1,
    _load_model,
    _predicted_controls,
    _room_iou,
    _safe_fpeval,
    _support_iou,
)

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.paired_bootstrap import paired_delta
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.geometry_heads import wall_junction_targets
from floorflow.heads.raster_control_bank import (
    RasterDenseQueryRoomBlend,
    RasterInstanceClassRoutedAdapter,
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.heads.structure_mita_router import structure_route_metrics
from floorflow.objectives.control_repair import (
    masked_room_loss,
    room_index_target,
    structure_losses,
)
from floorflow.objectives.structure_refinement import (
    canonical_plan_proxy_loss,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.training.raster_structure_bridge_cli import (
    parse_args,
    validate_args,
)
from floorflow.training.raster_structure_bridge_data import (
    RasterPredictedControlDataset,
    bridge_controls,
)

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def evaluate_bridge(
    model,
    raster_room,
    loader,
    taxonomy,
    args,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    raster_room.eval()
    device = torch.device(args.device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    room_count = len(taxonomy.room_ids)
    rows = []
    for target, footprint, _floor_ids, names, image in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        image = image.to(device)
        predicted = _predicted_controls(
            args.predicted_root,
            "val",
            names,
            args,
            device,
        )
        raster_logits = F.interpolate(
            raster_room(footprint, control_image=image),
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        controls = bridge_controls(predicted, raster_logits)
        output = model(
            controls["room_logits"],
            controls["structure"],
            footprint,
            torch.zeros_like(controls["room_logits"][:, :1]),
            torch.zeros(
                (
                    target.shape[0],
                    1,
                    args.structure_size,
                    args.structure_size,
                ),
                device=device,
            ),
            program=controls["program"],
            repair_room=not args.structure_only,
        )
        labels = room_ids[output["room_logits_full"].argmax(dim=1)]
        labels = F.interpolate(
            labels[:, None].float(),
            size=target.shape[-2:],
            mode="nearest",
        )[:, 0].long()
        labels *= footprint.long()
        structure = output["structure"]
        route_metrics = structure_route_metrics(
            structure.get("route_diagnostics")
        )
        wall = F.interpolate(
            structure["wall_logits"].sigmoid(),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0] >= 0.5
        opening_type = F.interpolate(
            structure["opening_type_logits"],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        opening = opening_ids[opening_type] * (
            F.interpolate(
                structure["opening_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0] >= 0.5
        )
        visible = LayeredFloorplan(
            room_labels=labels,
            wall=wall.long(),
            opening=opening,
            furniture=torch.zeros_like(labels),
            footprint=footprint,
            taxonomy=taxonomy,
        ).visible_labels()
        clean = coarse_structure_control(
            target,
            footprint,
            taxonomy,
            size=args.structure_size,
        )
        junction_target = degree_target = None
        if "junction_logits" in structure:
            structure_footprint = F.interpolate(
                footprint[:, None].float(),
                size=(args.structure_size, args.structure_size),
                mode="nearest",
            )[:, 0]
            junction_target, degree_target = wall_junction_targets(
                clean[:, room_count : room_count + 1],
                structure_footprint,
            )
        for index in range(target.shape[0]):
            labels_np = labels[index].cpu().numpy()
            target_np = target[index].cpu().numpy()
            room_support = torch.isin(labels[index], room_ids)
            target_support = torch.isin(target[index], room_ids)
            suite = room_metric_suite(
                labels_np,
                target_np,
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=room_support.cpu().numpy(),
                predicted_instance_support=room_support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            fpeval = _safe_fpeval(
                visible[index].cpu().numpy(),
                target_np,
                taxonomy,
            )
            structural = structural_legality_metrics(
                visible[index].cpu().numpy(),
                target_np,
                taxonomy,
            )
            row = {
                    "room_macro_iou": _room_iou(
                        labels_np,
                        target_np,
                        taxonomy.room_ids,
                    ),
                    "room_support_iou": _support_iou(
                        labels_np,
                        target_np,
                        taxonomy.room_ids,
                    ),
                    "wall_f1": _binary_f1(
                        structure["wall_logits"][index, 0].sigmoid(),
                        clean[index, room_count],
                        torch.ones_like(
                            structure["wall_logits"][index, 0],
                            dtype=torch.bool,
                        ),
                    ),
                    "opening_f1": _binary_f1(
                        structure["opening_logits"][index, 0].sigmoid(),
                        clean[index, room_count + 1],
                        torch.ones_like(
                            structure["opening_logits"][index, 0],
                            dtype=torch.bool,
                        ),
                    ),
                    "validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "watertight": float(
                        fpeval["prediction_validity"]["watertight"]
                    ),
                    "instance_geometric_iou": float(
                        instance["matched_geometric_iou"]
                    ),
                    "instance_room_f1": float(
                        instance["instance_f1_at_threshold"]
                    ),
                    "instance_type_accuracy": float(
                        instance["matched_type_accuracy"]
                    ),
                    "room_closure_rate": float(
                        structural["room_closure_rate"]
                    ),
                    **route_metrics,
            }
            if junction_target is not None and degree_target is not None:
                row["junction_f1"] = _binary_f1(
                    structure["junction_logits"][index, 0].sigmoid(),
                    junction_target[index, 0],
                    torch.ones_like(
                        junction_target[index, 0],
                        dtype=torch.bool,
                    ),
                )
                valid_junction = degree_target[index] > 0
                row["junction_degree_accuracy"] = (
                    float(
                        (
                            structure["junction_degree_logits"][
                                index
                            ].argmax(dim=0)[valid_junction]
                            == degree_target[index][valid_junction]
                        ).float().mean()
                    )
                    if valid_junction.any()
                    else 0.0
                )
            rows.append(row)
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    taxonomy = CubiCasaFloorplanDataset(
        args.data_root,
        "val",
        cache_root=args.cache_root,
        limit=1,
    ).taxonomy
    train = RasterPredictedControlDataset(
        args.predicted_root,
        "train",
        args.cache_root,
        args.data_root,
        limit=args.limit_train,
        image_size=args.image_size,
    )
    validation = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        limit=args.limit_val,
        image_size=args.image_size,
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(validation, shuffle=False, **common)
    model = _load_model(args, taxonomy, device)
    if args.room_query_checkpoint is None:
        raster_room = load_raster_room_model(
            args.raster_room_checkpoint,
            taxonomy,
            device,
        )
    elif args.room_query_blend is not None:
        raster_room = RasterDenseQueryRoomBlend(
            load_raster_room_model(
                args.raster_room_checkpoint,
                taxonomy,
                device,
            ),
            load_raster_instance_control_model(
                args.room_query_checkpoint,
                taxonomy,
                device,
            ),
            args.room_query_blend,
        ).to(device)
    else:
        raster_room = RasterInstanceClassRoutedAdapter(
            load_raster_instance_control_model(
                args.room_query_checkpoint,
                taxonomy,
                device,
            ),
            threshold=args.room_mask_threshold,
            strength=args.room_mask_strength,
        ).to(device)
    raster_room.requires_grad_(False)
    baseline, baseline_rows = evaluate_bridge(
        model,
        raster_room,
        val_loader,
        taxonomy,
        args,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    best_score = -1.0
    best_state = None
    best_rows = baseline_rows
    history = []
    room_tensor = torch.tensor(taxonomy.room_ids, device=device)
    opening_tensor = torch.tensor(taxonomy.opening_ids, device=device)
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        totals = {}
        seen = 0
        for target, footprint, _floor_ids, names, image in train_loader:
            target = target.to(device)
            footprint = footprint.to(device)
            image = image.to(device)
            predicted = _predicted_controls(
                args.predicted_root,
                "train",
                names,
                args,
                device,
            )
            with torch.no_grad():
                if (
                    isinstance(raster_room, RasterDenseQueryRoomBlend)
                    and args.room_blend_jitter > 0.0
                ):
                    raster_room.blend = min(
                        1.0,
                        max(
                            0.0,
                            float(args.room_query_blend)
                            + random.uniform(
                                -args.room_blend_jitter,
                                args.room_blend_jitter,
                            ),
                        ),
                    )
                raster_logits = F.interpolate(
                    raster_room(footprint, control_image=image),
                    size=(args.room_size, args.room_size),
                    mode="bilinear",
                    align_corners=False,
                )
            controls = bridge_controls(predicted, raster_logits)
            output = model(
                controls["room_logits"],
                controls["structure"],
                footprint,
                torch.zeros_like(controls["room_logits"][:, :1]),
                torch.zeros(
                    (
                        target.shape[0],
                        1,
                        args.structure_size,
                        args.structure_size,
                    ),
                    device=device,
                ),
                program=controls["program"],
                repair_room=not args.structure_only,
            )
            if args.structure_only:
                room_loss = output["room_logits_full"].sum() * 0.0
                room_terms = {
                    "room_ce": room_loss.detach(),
                    "room_dice": room_loss.detach(),
                }
            else:
                room_loss, room_terms = masked_room_loss(
                    output["room_logits_full"],
                    room_index_target(target, room_tensor),
                    torch.ones_like(target[:, None], dtype=torch.bool),
                )
            clean = coarse_structure_control(target, footprint, taxonomy, size=args.structure_size)
            structure_loss, structure_terms = structure_losses(
                output["structure"],
                clean,
                torch.ones(
                    (
                        clean.shape[0],
                        clean.shape[-2],
                        clean.shape[-1],
                    ),
                    device=device,
                    dtype=torch.bool,
                ),
                len(taxonomy.room_ids),
                opening_tensor,
                centerline_weight=args.centerline_weight,
            )
            loss = room_loss + structure_loss
            relation_terms = {}
            if (
                args.structure_contract == "typed_v2"
                and args.structure_relation_weight > 0.0
            ):
                relation_loss, relation_terms = (
                    canonical_plan_proxy_loss(
                        output["structure"],
                        output["room_logits_full"],
                        clean,
                        target,
                        footprint,
                        taxonomy,
                        room_count=len(taxonomy.room_ids),
                        opening_host_weight=args.structure_opening_host_weight,
                        room_edge_weight=args.structure_room_edge_weight,
                        junction_weight=args.structure_junction_weight,
                        boundary_weight=args.structure_boundary_weight,
                    )
                )
                loss = (
                    loss
                    + args.structure_relation_weight * relation_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in {
                **room_terms,
                **structure_terms,
                **relation_terms,
            }.items():
                totals[key] = (
                    totals.get(key, 0.0)
                    + float(value) * target.shape[0]
                )
            seen += target.shape[0]
        scheduler.step()
        if isinstance(raster_room, RasterDenseQueryRoomBlend):
            raster_room.blend = float(args.room_query_blend)
        summary, rows = evaluate_bridge(
            model,
            raster_room,
            val_loader,
            taxonomy,
            args,
        )
        score = (
            summary["room_macro_iou"]
            + 0.25 * summary["wall_f1"]
            + 0.25 * summary["opening_f1"]
            + 0.1 * summary["watertight"]
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value / max(seen, 1)
                for key, value in totals.items()
            },
            **summary,
            "selection_score": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best_rows = rows
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
        best_summary, best_rows = evaluate_bridge(
            model,
            raster_room,
            val_loader,
            taxonomy,
            args,
        )
    else:
        best_summary = baseline
        best_rows = baseline_rows
    paired = {
        metric: paired_delta(
            best_rows,
            baseline_rows,
            metric,
            iterations=args.bootstrap,
            seed=args.seed + 24000,
        )
        for metric in (
            "room_macro_iou",
            "room_support_iou",
            "wall_f1",
            "opening_f1",
            "validity",
            "watertight",
            "instance_geometric_iou",
            "instance_room_f1",
            "instance_type_accuracy",
            "room_closure_rate",
        )
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_json(),
            "config": vars(args) | {"device": str(device)},
            "metrics": best_summary,
        },
        checkpoint,
    )
    report = {
        "schema_version": "raster-predicted-control-bridge-v1",
        "scope": "raster-conditioned reconstruction",
        "target_used_at_inference": False,
        "room_labels_in_native_input": False,
        "split": "val",
        "cases": len(validation),
        "config": {**vars(args), "device": str(device)},
        "baseline": baseline,
        "best": best_summary,
        "paired": paired,
        "history": history,
        "checkpoint": (
            str(checkpoint.relative_to(ROOT))
            if checkpoint.is_absolute()
            else str(checkpoint)
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report), "best": best_summary}, indent=2))


__all__ = [
    "RasterPredictedControlDataset",
    "bridge_controls",
    "evaluate_bridge",
    "main",
]

if __name__ == "__main__":
    main()
