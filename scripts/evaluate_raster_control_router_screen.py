#!/usr/bin/env python3
"""Screen target-free routing and topology repair for raster controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_raster_predicted_control_bridge import (
    RasterPredictedControlDataset,
    bridge_controls,
)
from train_residential_predicted_refiner import (
    _binary_f1,
    _predicted_controls,
    _safe_fpeval,
    _support_iou,
)

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    watertight_diagnostics,
)
from floorflow.evaluation.room_corner_metrics import room_corner_angle_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.evaluation.structural_legality import structural_legality_metrics
from floorflow.geometry.core import compose_known_exterior
from floorflow.geometry.wall_graph_readout import raster_labels_to_wall_plan
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.joint_control_repair import JointControlRepair
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.control_reliability_router import route_room_control
from floorflow.inference.room_polygon_readout import vectorize_room_masks
from floorflow.inference.room_repair import apply_floorplan_repairs
from floorflow.inference.structure_compositor import (
    StructureReadoutPolicy,
    decode_structure_layers,
)
from floorflow.representation.layered_floorplan import LayeredFloorplan

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predicted-root",
        type=Path,
        default=ROOT / "outputs/control_contracts/model_predicted_full_v1",
    )
    parser.add_argument(
        "--structure-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_structure_bridge_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_predicted_control_bridge_screen256_v1/best.pt",
    )
    parser.add_argument(
        "--raster-room-checkpoint",
        type=Path,
        default=ROOT / "outputs/train/raster_room_control_native_full_v1/best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data/cubicasa5k")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "outputs/cache/cubicasa5k_full_v2",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/raster_control_router_screen256_v1.json",
    )
    return parser.parse_args()


def _structure_outputs(
    model,
    predicted,
    raster_logits,
    footprint,
    args,
    *,
    repair_room: bool,
):
    controls = bridge_controls(predicted, raster_logits)
    return model(
        controls["room_logits"],
        controls["structure"],
        footprint,
        torch.zeros_like(controls["room_logits"][:, :1]),
        torch.zeros(
            (
                footprint.shape[0],
                1,
                args.structure_size,
                args.structure_size,
            ),
            device=footprint.device,
        ),
        program=controls["program"],
        repair_room=repair_room,
    )


def _decode_layers(
    room_logits,
    structure,
    footprint,
    taxonomy,
    policy: StructureReadoutPolicy | None = None,
):
    layers, _diagnostics = decode_structure_layers(
        room_logits,
        structure,
        footprint,
        taxonomy,
        policy=policy,
    )
    return (
        layers,
        layers.room_labels,
        layers.wall,
        layers.opening,
    )


def _postprocess(
    layers: LayeredFloorplan,
    variant: str,
    taxonomy,
) -> torch.Tensor:
    labels = layers.visible_labels()
    footprint = layers.footprint
    if variant == "none":
        return labels
    if variant == "exterior":
        return compose_known_exterior(
            labels,
            footprint,
            opening_ids=taxonomy.opening_ids,
        )
    if variant == "repair":
        repaired = torch.stack(
            [
                torch.from_numpy(
                    apply_floorplan_repairs(
                        row.cpu().numpy(),
                        mask.cpu().numpy().astype(np.uint8),
                        taxonomy,
                    )
                ).to(labels.device)
                for row, mask in zip(labels, footprint, strict=True)
            ]
        )
        return compose_known_exterior(
            repaired,
            footprint,
            opening_ids=taxonomy.opening_ids,
        )
    if variant == "partition":
        rooms = layers.room_labels * footprint.long()
        wall = partition_wall_mask(rooms, radius=1) & footprint
        wall_support = F.max_pool2d(
            wall.float()[:, None],
            5,
            stride=1,
            padding=2,
        )[:, 0] > 0.5
        opening = torch.where(
            wall_support,
            layers.opening,
            torch.zeros_like(layers.opening),
        )
        partition_layers = LayeredFloorplan(
            room_labels=rooms,
            wall=wall.long(),
            opening=opening,
            furniture=layers.furniture,
            footprint=footprint,
            taxonomy=taxonomy,
        )
        return compose_known_exterior(
            partition_layers.visible_labels(),
            footprint,
            opening_ids=taxonomy.opening_ids,
        )
    if variant.startswith("vector_"):
        epsilon = float(variant.removeprefix("vector_"))
        return torch.stack(
            [
                torch.from_numpy(
                    vectorize_room_masks(
                        row.cpu().numpy(),
                        mask.cpu().numpy().astype(bool),
                        taxonomy,
                        epsilon=epsilon,
                    )
                ).to(labels.device)
                for row, mask in zip(labels, footprint, strict=True)
            ]
        )
    raise ValueError(f"unknown variant: {variant}")


def _metric_row(
    labels,
    target,
    footprint,
    taxonomy,
    clean_structure,
    room_count,
) -> dict[str, float]:
    room_ids = torch.tensor(taxonomy.room_ids, device=labels.device)
    support = torch.isin(labels, room_ids)
    target_support = torch.isin(target, room_ids)
    suite = room_metric_suite(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        footprint.cpu().numpy(),
        taxonomy,
        predicted_room_support=support.cpu().numpy(),
        predicted_instance_support=support.cpu().numpy(),
        target_room_support=target_support.cpu().numpy(),
    )
    instance = suite["matched_instance"]
    fpeval = _safe_fpeval(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        taxonomy,
    )
    fpeval_summary = fpeval.get("summary", {}).get("0.05", {})
    structural = structural_legality_metrics(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        taxonomy,
    )
    corner_angle = room_corner_angle_metrics(
        labels.cpu().numpy(),
        target.cpu().numpy(),
        taxonomy,
    )
    plan = raster_labels_to_wall_plan(
        labels.cpu().numpy(),
        taxonomy,
    )
    watertight = watertight_diagnostics(
        plan,
        taxonomy=taxonomy,
        labels=labels.cpu().numpy(),
        footprint=footprint.cpu().numpy(),
        tolerances=FEVPTolerances(),
    )
    values = {
        "room_macro_iou": float(
            suite["visible_semantic"]["room_macro_iou"]
        ),
        "room_support_iou": _support_iou(
            labels.cpu().numpy(),
            target.cpu().numpy(),
            taxonomy.room_ids,
        ),
        "instance_geometric_iou": float(
            instance["matched_geometric_iou"]
        ),
        "instance_room_f1": float(instance["instance_f1_at_threshold"]),
        "instance_type_accuracy": float(instance["matched_type_accuracy"]),
        "predicted_room_count": float(instance["predicted_instances"]),
        "target_room_count": float(instance["target_instances"]),
        "room_count_abs_error": float(instance["count_abs_error"]),
        "validity": float(fpeval["prediction_validity"]["valid"]),
        "watertight": float(fpeval["prediction_validity"]["watertight"]),
        "parsable": structural["parsable"],
        "room_closure_rate": structural["room_closure_rate"],
        "wall_continuity_rate": structural["wall_continuity_rate"],
        "self_intersection_rate": structural["self_intersection_rate"],
        "room_overlap_ratio": structural["room_overlap_ratio"],
        "out_of_boundary_ratio": structural["out_of_boundary_ratio"],
        "node_f1": structural["node_f1"],
        "edge_overlap": structural["edge_overlap"],
        "ged": structural["ged"],
        "watertight_diagnostic_rate": watertight["watertight_rate"],
        "hole_count": watertight["hole_count"],
        "dangling_endpoint_rate": watertight["dangling_endpoint_rate"],
        "edit_cost": float(fpeval_summary.get("edit_cost", 0.0)),
        "edit_cost_per_gt_wall": float(
            fpeval_summary.get("edit_cost_per_gt_wall", 0.0)
        ),
        **corner_angle,
    }
    wall_id = taxonomy.class_id("wall")
    opening_ids = torch.tensor(taxonomy.opening_ids, device=labels.device)
    if clean_structure.shape[-2:] != labels.shape[-2:]:
        clean_structure = F.interpolate(
            clean_structure[None].float(),
            size=labels.shape[-2:],
            mode="nearest",
        )[0].long()
    values["wall_f1"] = _binary_f1(
        (labels == wall_id).float(),
        clean_structure[room_count],
        torch.ones_like(labels, dtype=torch.bool),
    )
    values["opening_f1"] = _binary_f1(
        torch.isin(labels, opening_ids).float(),
        clean_structure[room_count + 1],
        torch.ones_like(labels, dtype=torch.bool),
    )
    return values


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        limit=args.limit,
        offset=args.offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    structure_model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(device)
    structure_payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure_model.load_state_dict(structure_payload["model"])
    joint_model = JointControlRepair(
        len(taxonomy.room_ids),
        len(taxonomy.opening_ids),
        room_base_channels=args.room_base_channels,
        structure_base_channels=args.structure_base_channels,
    ).to(device)
    joint_payload = torch.load(
        args.joint_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    joint_model.load_state_dict(joint_payload["model"])
    raster_room = load_raster_room_model(
        args.raster_room_checkpoint,
        taxonomy,
        device,
    )
    variants = (
        "none",
        "exterior",
        "repair",
        "partition",
        "vector_1.5",
        "vector_3.0",
        "vector_5.0",
    )
    router_settings = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
    )
    arms = {
        f"{branch}/{variant}": []
        for branch in ("structure", "joint")
        for variant in variants
    }
    arms.update(
        {
            f"router_a{agreement:.2f}_m{margin:.2f}/{variant}": []
            for agreement, margin in router_settings
            for variant in variants
        }
    )
    diagnostics = {}
    room_count = len(taxonomy.room_ids)
    with torch.inference_mode():
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
            raw_logits = F.interpolate(
                raster_room(footprint, control_image=image),
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            structure_out = _structure_outputs(
                structure_model,
                predicted,
                raw_logits,
                footprint,
                args,
                repair_room=False,
            )
            joint_out = _structure_outputs(
                joint_model,
                predicted,
                raw_logits,
                footprint,
                args,
                repair_room=True,
            )
            structure_layers, raw_rooms, raw_wall, raw_opening = _decode_layers(
                raw_logits,
                structure_out["structure"],
                footprint,
                taxonomy,
            )
            joint_layers, joint_rooms, joint_wall, joint_opening = _decode_layers(
                joint_out["room_logits_full"],
                joint_out["structure"],
                footprint,
                taxonomy,
            )
            clean = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            for branch, layers in (
                ("structure", structure_layers),
                ("joint", joint_layers),
            ):
                for variant in variants:
                    labels = _postprocess(layers, variant, taxonomy)
                    for index, name in enumerate(names):
                        arms[f"{branch}/{variant}"].append(
                            {
                                "case": str(name),
                                **_metric_row(
                                    labels[index],
                                    target[index],
                                    footprint[index],
                                    taxonomy,
                                    clean[index],
                                    room_count,
                                ),
                            }
                        )
            for agreement, margin in router_settings:
                routed_primary = F.interpolate(
                    raw_logits,
                    size=joint_out["room_logits_full"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                routed = route_room_control(
                    routed_primary,
                    joint_out["room_logits_full"],
                    footprint,
                    minimum_agreement=agreement,
                    confidence_margin=margin,
                )
                routed_rooms = torch.where(
                    routed.selected_candidate[:, None, None],
                    joint_rooms,
                    raw_rooms,
                )
                routed_wall = torch.where(
                    routed.selected_candidate[:, None, None],
                    joint_wall,
                    raw_wall,
                )
                routed_opening = torch.where(
                    routed.selected_candidate[:, None, None],
                    joint_opening,
                    raw_opening,
                )
                routed_layers = LayeredFloorplan(
                    room_labels=routed_rooms,
                    wall=routed_wall.long(),
                    opening=routed_opening,
                    furniture=torch.zeros_like(routed_rooms),
                    footprint=footprint,
                    taxonomy=taxonomy,
                )
                key = f"router_a{agreement:.2f}_m{margin:.2f}"
                diagnostics[key] = {
                    "selected_fraction": float(
                        routed.selected_candidate.float().mean()
                    ),
                    "agreement_mean": float(routed.agreement.mean()),
                    "confidence_mean": float(routed.confidence.mean()),
                }
                for variant in variants:
                    labels = _postprocess(
                        routed_layers,
                        variant,
                        taxonomy,
                    )
                    for index, name in enumerate(names):
                        arms[f"{key}/{variant}"].append(
                            {
                                "case": str(name),
                                **_metric_row(
                                    labels[index],
                                    target[index],
                                    footprint[index],
                                    taxonomy,
                                    clean[index],
                                    room_count,
                                ),
                            }
                        )
    summaries = {
        arm: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
            if key != "case"
        }
        for arm, rows in arms.items()
    }
    report = {
        "schema_version": "raster-control-router-screen-v1",
        "scope": "mechanism screening only",
        "split": f"CubiCasa val{len(dataset)} subset",
        "validation_offset": args.offset,
        "train_cases_used_by_branch_models": 256,
        "test_opened": False,
        "target_used_at_inference": False,
        "raster_room_checkpoint": str(args.raster_room_checkpoint),
        "structure_checkpoint": str(args.structure_checkpoint),
        "joint_checkpoint": str(args.joint_checkpoint),
        "router_diagnostics": diagnostics,
        "summary": summaries,
        "promotion_allowed": False,
        "note": "Branch checkpoints were selected on this fixed validation subset; use this screen only to choose the next mechanism.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
