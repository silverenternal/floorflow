#!/usr/bin/env python3
"""Evaluate stage-2 refiners on self-predicted distribution labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.candidate_delta import shared_variant_delta
from floorflow.evaluation.distribution_overlay_metrics import constraint_metrics
from floorflow.evaluation.fpeval_bridge import (
    fpeval_raster_metrics,
    prepare_target_plan,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.coarse_furniture_refiner import (
    CoarseFurnitureRefiner,
    coarse_furniture_control,
)
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.coarse_support_refiner import (
    CoarseSupportRefiner,
    coarse_support_control,
)
from floorflow.heads.structured_objects import (
    object_group_metrics,
)
from floorflow.inference.distribution_overlay import DistributionOverlaySettings
from floorflow.inference.overlay_decoder import decode_overlay
from floorflow.inference.recursive_room_prior import (
    load_graph_adapter,
    load_recursion_policy,
    predict_recursive_room_prior,
)
from floorflow.inference.room_repair import fill_room_support
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.representation.layered_floorplan import LayeredFloorplan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--furniture-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_geometry_v1"),
    )
    parser.add_argument(
        "--retrieval-weights",
        type=Path,
        default=Path("outputs/reports/a1_retrieval_weights_calib191.json"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--room-segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--recursion-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--recursive-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--recursive-graph-blend-scale", type=float, default=1.0)
    parser.add_argument("--room-prior-weight", type=float, default=0.0)
    parser.add_argument("--room-prior-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--retrieval-prior-weight", type=float, default=1.0)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--wall-threshold", type=float, default=0.6)
    parser.add_argument("--furniture-threshold", type=float, default=0.5)
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("raw", "refined", "fused", "hybrid", "objects_only", "support_completed"),
        default=("raw", "refined", "fused", "hybrid", "support_completed"),
    )
    return parser.parse_args()


def load_refiners(args, taxonomy, device):
    payload = torch.load(args.structure_checkpoint, map_location="cpu", weights_only=False)
    structure = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    structure.load_state_dict(payload["model"])
    payload = torch.load(args.furniture_checkpoint, map_location="cpu", weights_only=False)
    furniture = CoarseFurnitureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        len(taxonomy.furniture_ids),
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    furniture.load_state_dict(payload["model"])
    payload = torch.load(args.support_checkpoint, map_location="cpu", weights_only=False)
    support = CoarseSupportRefiner(
        len(taxonomy.room_ids) + 4,
        base_channels=int(payload["config"].get("base_channels", 32)),
    ).to(device)
    support.load_state_dict(payload["model"])
    return structure.eval(), furniture.eval(), support.eval()


def geometry_path(root, split, name):
    floor = Path(name.replace("__", "/"))
    return (
        root
        / split
        / floor.parent
        / f"{floor.name.replace('-', '_')}.geometry.npz"
    )


def support_iou(prediction, target, footprint):
    prediction = np.asarray(prediction, dtype=bool) & footprint
    target = np.asarray(target, dtype=bool) & footprint
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(
        args.distribution_checkpoint,
        device,
    )
    structure_model, furniture_model, support_model = load_refiners(
        args,
        taxonomy,
        device,
    )
    room_segmenter = None
    if args.room_segmenter_checkpoint is not None:
        room_segmenter, _segmenter_config = load_room_segmenter_prior(
            args.room_segmenter_checkpoint,
            taxonomy,
            device,
        )
    recursion_policy = (
        load_recursion_policy(args.recursion_policy_checkpoint, device)
        if args.recursion_policy_checkpoint is not None
        else None
    )
    graph_adapter = (
        load_graph_adapter(
            args.recursive_graph_checkpoint,
            device,
            blend_scale=args.recursive_graph_blend_scale,
        )
        if args.recursive_graph_checkpoint is not None
        else None
    )
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    use_retrieval = float(args.retrieval_prior_weight) > 0.0
    memory = (
        ContourPartitionMemory.from_cache(args.cache_root, top_k=1)
        if use_retrieval
        else None
    )
    retrieval_weights = (
        torch.tensor(
            json.loads(args.retrieval_weights.read_text(encoding="utf-8"))[
                "weights"
            ],
            device=device,
        )
        if use_retrieval
        else None
    )
    settings = DistributionOverlaySettings(opening_element_weight=0.0)
    selected_variants = tuple(dict.fromkeys(args.variants))
    selected_variant_set = set(selected_variants)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device)
    rows = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        style_ids = torch.zeros(footprint.shape[0], dtype=torch.long, device=device)
        room_prior_logits = None
        if room_segmenter is not None:
            direct_prior = predict_room_segmenter_prior(
                room_segmenter,
                footprint,
            )
            room_prior_logits = direct_prior
            if recursion_policy is not None:
                recursive = predict_recursive_room_prior(
                    recursion_policy,
                    room_segmenter,
                    footprint,
                    context_fraction=0.75,
                    global_logits=direct_prior,
                    recursive_blend=0.1,
                    graph_adapter=graph_adapter,
                )
                room_prior_logits = torch.where(
                    (recursive.action == 0)[:, None, None, None],
                    direct_prior,
                    recursive.logits,
                )
        decoded = decode_overlay(
            model,
            memory,
            retrieval_weights,
            footprint,
            taxonomy,
            floor_ids=floor_ids.to(device),
            style_ids=style_ids,
            retrieval_prior_weight=args.retrieval_prior_weight,
            room_prior_logits=room_prior_logits,
            room_prior_weight=args.room_prior_weight,
            room_prior_confidence_threshold=(
                args.room_prior_confidence_threshold
            ),
            settings=settings,
        )
        rooms = decoded["layers"].room_labels
        structure = structure_model(
            coarse_structure_control(rooms, footprint, taxonomy, size=128)
        )
        opening = opening_ids[
            F.interpolate(
                structure["opening_type_logits"],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
        ]
        opening = torch.where(
            F.interpolate(
                structure["opening_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0] >= float(args.opening_threshold),
            opening,
            torch.zeros_like(opening),
        )
        wall = (
            F.interpolate(
                structure["wall_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0] >= float(args.wall_threshold)
        ).long() & footprint[:, 0].long()
        furniture = furniture_model(
            coarse_furniture_control(rooms, footprint, taxonomy, size=192),
            output_size=target.shape[-2:],
        )
        furniture_labels = furniture_ids[
            F.interpolate(
                furniture["class_logits"],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
        ]
        furniture_labels = torch.where(
            F.interpolate(
                furniture["foreground_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0] >= float(args.furniture_threshold),
            furniture_labels,
            torch.zeros_like(furniture_labels),
        )
        predicted_support = support_model(
            coarse_support_control(rooms, footprint, taxonomy, size=192),
            output_size=target.shape[-2:],
        ).sigmoid()[:, 0] >= float(args.support_threshold)
        refined = LayeredFloorplan(
            room_labels=rooms,
            wall=wall,
            opening=opening,
            furniture=furniture_labels,
            footprint=footprint[:, 0],
            taxonomy=taxonomy,
        )
        fused_wall = decoded["layers"].wall | refined.wall
        fused_layers = LayeredFloorplan(
            room_labels=refined.room_labels,
            wall=fused_wall.long(),
            opening=refined.opening,
            furniture=refined.furniture,
            footprint=footprint[:, 0],
            taxonomy=taxonomy,
        )
        raw_labels = decoded["layers"].visible_labels()
        refined_labels = refined.visible_labels()
        fused_labels = fused_layers.visible_labels()
        hybrid_layers = LayeredFloorplan(
            room_labels=decoded["layers"].room_labels,
            wall=fused_wall.long(),
            opening=refined.opening,
            furniture=refined.furniture,
            footprint=footprint[:, 0],
            taxonomy=taxonomy,
        )
        hybrid_labels = hybrid_layers.visible_labels()
        objects_only_labels = LayeredFloorplan(
            room_labels=decoded["layers"].room_labels,
            wall=decoded["layers"].wall,
            opening=refined.opening,
            furniture=refined.furniture,
            footprint=footprint[:, 0],
            taxonomy=taxonomy,
        ).visible_labels()
        for index in range(target.shape[0]):
            case_target = target[index : index + 1]
            inside = footprint[index, 0].cpu().numpy().astype(bool)
            with np.load(
                geometry_path(args.geometry_root, args.split, names[index])
            ) as geometry:
                target_support = geometry["room_support"] > 0
            predicted_rooms = refined.room_labels[index].cpu().numpy() > 0
            refined_support = predicted_support[index].cpu().numpy()
            completed_room_np = fill_room_support(
                refined.room_labels[index].cpu().numpy(),
                inside,
                taxonomy,
                support_mask=refined_support,
            )
            completed_layers = LayeredFloorplan(
                room_labels=torch.as_tensor(
                    completed_room_np,
                    device=footprint.device,
                    dtype=torch.long,
                ).reshape(1, *inside.shape),
                wall=fused_wall[index : index + 1],
                opening=refined.opening[index : index + 1],
                furniture=refined.furniture[index : index + 1],
                footprint=footprint[index : index + 1, 0],
                taxonomy=taxonomy,
            )
            completed_labels = completed_layers.visible_labels()
            raw = object_group_metrics(
                {
                    "wall": decoded["layers"].wall[index : index + 1],
                    "opening": decoded["layers"].opening[index : index + 1],
                    "furniture": decoded["layers"].furniture[index : index + 1],
                },
                case_target,
                taxonomy,
            )
            new = object_group_metrics(
                {
                    "wall": refined.wall[index : index + 1],
                    "opening": refined.opening[index : index + 1],
                    "furniture": refined.furniture[index : index + 1],
                },
                case_target,
                taxonomy,
            )
            new["support_iou"] = support_iou(
                refined_support,
                target_support,
                inside,
            )
            raw["support_iou"] = support_iou(
                predicted_rooms,
                target_support,
                inside,
            )
            fused = object_group_metrics(
                {
                    "wall": fused_wall[index : index + 1],
                    "opening": refined.opening[index : index + 1],
                    "furniture": refined.furniture[index : index + 1],
                },
                case_target,
                taxonomy,
            )
            fused["support_iou"] = new["support_iou"]
            case_reports = {}
            case_labels_map = {
                "raw": raw_labels,
                "refined": refined_labels,
                "fused": fused_labels,
                "hybrid": hybrid_labels,
                "objects_only": objects_only_labels,
                "support_completed": completed_labels,
            }
            target_plan = prepare_target_plan(
                case_target[0].cpu().numpy(),
                taxonomy,
            )
            for name in args.variants:
                labels = case_labels_map[name]
                report = dict({
                    "raw": raw,
                    "refined": new,
                    "fused": fused,
                    "hybrid": dict(fused),
                    "objects_only": dict(fused),
                    "support_completed": dict(fused),
                }[name])
                case_labels = (
                    labels[index] if labels.shape[0] > 1 else labels[0]
                )
                fpeval = fpeval_raster_metrics(
                    case_labels.cpu().numpy(),
                    case_target[0].cpu().numpy(),
                    taxonomy,
                    tolerance_fractions=(0.05,),
                    target_plan=target_plan,
                )
                summary = fpeval["summary"]["0.05"]
                room_ids = torch.tensor(
                    taxonomy.room_ids,
                    device=labels.device,
                )
                room_support = torch.isin(case_labels, room_ids)
                target_support = torch.isin(case_target[0], room_ids)
                suite = room_metric_suite(
                    case_labels.cpu().numpy(),
                    case_target[0].cpu().numpy(),
                    footprint[index, 0].cpu().numpy(),
                    taxonomy,
                    predicted_room_support=room_support.cpu().numpy(),
                    predicted_instance_support=room_support.cpu().numpy(),
                    target_room_support=target_support.cpu().numpy(),
                )
                instance = suite["matched_instance"]
                report.update(
                    {
                        "room_macro_iou": suite["visible_semantic"][
                            "room_macro_iou"
                        ],
                        "instance_geometric_iou": instance[
                            "matched_geometric_iou"
                        ],
                        "instance_room_f1": instance[
                            "instance_f1_at_threshold"
                        ],
                        "instance_type_accuracy": instance[
                            "matched_type_accuracy"
                        ],
                        "class_constrained_iou": instance[
                            "matched_class_constrained_iou"
                        ],
                        "room_support_iou": suite[
                            "occlusion_aware_support"
                        ]["room_support_iou"],
                        "room_component_gap": abs(
                            float(instance["predicted_instances"])
                            - float(instance["target_instances"])
                        ),
                        "fpeval_room_face_f1": summary["room_f1"],
                        "fpeval_wall_f1": summary["wall_f1"],
                        "fpeval_opening_f1": summary["opening_f1"],
                        "fpeval_edit_cost": summary["edit_cost"],
                        "fpeval_validity": float(
                            fpeval["prediction_validity"]["valid"]
                        ),
                        "fpeval_watertight": float(
                            fpeval["prediction_validity"]["watertight"]
                        ),
                    }
                )
                case_reports[name] = report
            explicit_room_variants = {
                "hybrid": hybrid_labels,
                "objects_only": objects_only_labels,
                "support_completed": completed_labels,
            }
            for name in selected_variant_set.intersection(explicit_room_variants):
                case_labels = explicit_room_variants[name][0]
                room_fpeval = fpeval_raster_metrics(
                    case_labels.cpu().numpy(),
                    case_target[0].cpu().numpy(),
                    taxonomy,
                    tolerance_fractions=(0.05,),
                    explicit_prediction_rooms=True,
                    explicit_target_rooms=True,
                    target_plan=target_plan,
                )
                case_reports[name]["fpeval_room_polygon_f1"] = room_fpeval[
                    "summary"
                ]["0.05"]["room_f1"]
                case_reports[name].update(
                    constraint_metrics(
                        case_labels,
                        case_target[0],
                        footprint[index, 0],
                        fused_wall[index],
                        refined.opening[index],
                        refined.furniture[index],
                        taxonomy,
                    )
                )
            rows.append(case_reports)
    summary = {
        name: {
            key: float(np.mean([row[name][key] for row in rows]))
            for key in rows[0][name]
        }
        for name in selected_variants
    }
    report = {
        "cases": len(rows),
        "selection": "predicted overlay labels; no target-derived control",
        "distribution_checkpoint": str(args.distribution_checkpoint),
        "room_segmenter_checkpoint": None if args.room_segmenter_checkpoint is None else str(args.room_segmenter_checkpoint),
        "recursive_graph_checkpoint": None if args.recursive_graph_checkpoint is None else str(args.recursive_graph_checkpoint),
        "room_prior_weight": args.room_prior_weight,
        "retrieval_prior_weight": args.retrieval_prior_weight,
        "thresholds": {
            "opening": args.opening_threshold,
            "wall": args.wall_threshold,
            "furniture": args.furniture_threshold,
            "support": args.support_threshold,
        },
        "variants": selected_variants,
        "metrics": summary,
        "delta": shared_variant_delta(summary, "refined"),
        "fused_delta": shared_variant_delta(summary, "fused"),
        "hybrid_delta": shared_variant_delta(summary, "hybrid"),
        "objects_only_delta": shared_variant_delta(summary, "objects_only"),
        "support_completed_delta": shared_variant_delta(
            summary,
            "support_completed",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
