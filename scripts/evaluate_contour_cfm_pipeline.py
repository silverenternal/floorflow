#!/usr/bin/env python3
"""Evaluate the contour-only MiTA-CFM + refinement pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.flow.guided_control import (
    guided_source_from_labels,
    sample_guided_floorplans,
)
from floorflow.flow.mask_policies import AttributeMaskPolicy, make_mask_policy
from floorflow.heads.coarse_furniture_refiner import CoarseFurnitureRefiner
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.coarse_structure_refiner import CoarseStructureRefiner
from floorflow.heads.coarse_support_refiner import CoarseSupportRefiner
from floorflow.heads.geometry_heads import structural_candidate_mask
from floorflow.heads.pixel_distribution import PixelDistributionHead
from floorflow.heads.room_program import gate_room_labels_to_program
from floorflow.heads.structured_objects import object_group_metrics, partition_wall_mask
from floorflow.inference.cached_contour_controls import load_cached_contour_controls
from floorflow.inference.contour_controls import controls_from_target_channels, predict_contour_controls
from floorflow.model.distribution_initializer import blend_distribution_state, distribution_initial_state
from floorflow.model.learned_mask_generator import LearnedMaskGenerator
from floorflow.model.sparse_control_attention import MaskedControlFlow, sample_sparse_guided_floorplans
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.training.distribution_integration import attach_distribution_controller
from floorflow.training.distribution_runtime import set_training_seed
from floorflow.training.masked_conditioning import prepare_masked_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--room-checkpoint", type=Path, required=True)
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--furniture-checkpoint", type=Path, required=True)
    parser.add_argument("--support-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(".data/cubicasa5k"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/cubicasa5k_full_v2"))
    parser.add_argument("--geometry-root", type=Path, default=Path("outputs/cache/cubicasa5k_geometry_v1"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--guided-control", action="store_true")
    parser.add_argument("--masked-control", action="store_true")
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--mask-policy", choices=("spatial", "attribute", "composite", "learned"), default="learned")
    parser.add_argument("--cfm-residual-weight", type=float, default=1.0)
    parser.add_argument("--distribution-checkpoint", type=Path, default=None)
    parser.add_argument("--distribution-blend", type=float, default=0.0)
    parser.add_argument("--disable-distribution-gate", action="store_true")
    parser.add_argument("--disable-room-instance-head", action="store_true")
    parser.add_argument("--bypass-room-refiner", action="store_true")
    parser.add_argument("--trinity-repair", action="store_true")
    parser.add_argument("--trinity-refine-steps", type=int, default=8)
    parser.add_argument("--trinity-refine-lr", type=float, default=1.0e-3)
    parser.add_argument("--control-source", choices=("aux", "sampled", "blend"), default="blend")
    parser.add_argument("--room-size", type=int, default=64)
    parser.add_argument("--structure-size", type=int, default=64)
    parser.add_argument("--furniture-size", type=int, default=96)
    parser.add_argument("--support-size", type=int, default=96)
    parser.add_argument("--wall-threshold", type=float, default=0.6)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--furniture-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--control-override-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_refiners(args, taxonomy, device):
    room_payload = torch.load(args.room_checkpoint, map_location="cpu", weights_only=False)
    room = CoarseLayoutRefiner(len(taxonomy.room_ids)).to(device)
    room.load_state_dict(room_payload["model"])
    room.eval()
    structure_payload = torch.load(args.structure_checkpoint, map_location="cpu", weights_only=False)
    structure = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(structure_payload["config"].get("base_channels", 32)),
    ).to(device)
    structure.load_state_dict(structure_payload["model"])
    structure.eval()
    furniture_payload = torch.load(args.furniture_checkpoint, map_location="cpu", weights_only=False)
    furniture = CoarseFurnitureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.furniture_ids) + 1,
        len(taxonomy.furniture_ids),
        base_channels=int(furniture_payload["config"].get("base_channels", 32)),
    ).to(device)
    furniture.load_state_dict(furniture_payload["model"])
    furniture.eval()
    support_payload = torch.load(args.support_checkpoint, map_location="cpu", weights_only=False)
    support = CoarseSupportRefiner(
        len(taxonomy.room_ids) + 4,
        base_channels=int(support_payload["config"].get("base_channels", 32)),
    ).to(device)
    support.load_state_dict(support_payload["model"])
    support.eval()
    return room, structure, furniture, support


def load_distribution_head(path, taxonomy, in_channels, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    output_weight = state.get("mix.3.weight")
    first_weight = state["mix.0.weight"]
    if output_weight is None:
        raise ValueError("distribution checkpoint has no classification head")
    head = PixelDistributionHead(
        int(in_channels),
        int(output_weight.shape[0]),
        hidden_channels=int(first_weight.shape[0]),
        room_ids=taxonomy.room_ids,
    ).to(device)
    head.load_state_dict(state)
    return head.eval()


def geometry_path(root: Path, split: str, name: str) -> Path:
    floor = Path(name.replace("__", "/"))
    return root / split / floor.parent / f"{floor.name.replace('-', '_')}.geometry.npz"


def support_iou(prediction, target, region):
    union = np.logical_or(prediction, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum() / union)


def safe_fpeval(prediction, target, taxonomy):
    try:
        return fpeval_raster_metrics(
            prediction,
            target,
            taxonomy,
            tolerance_fractions=(0.05,),
        )
    except (IndexError, ValueError):
        zero_summary = {
            "wall_f1": 0.0,
            "opening_f1": 0.0,
            "room_f1": 0.0,
        }
        return {
            "summary": {"0.05": zero_summary},
            "prediction_validity": {"valid": False, "watertight": False},
            "target_validity": {"watertight": True},
        }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    device = torch.device(args.device)
    stage1, config, taxonomy, _state = load_checkpoint_model(
        args.stage1_checkpoint,
        device,
        taxonomy=None,
    )
    payload = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    stage1 = attach_distribution_controller(
        stage1,
        taxonomy,
        head_state=payload.get("distribution_head"),
        gate_state=payload.get("distribution_gate"),
        hidden_channels=int(config.get("distribution_hidden", 96)),
        gate_hidden=int(config.get("distribution_gate_hidden", 64)),
    )
    if (
        args.disable_distribution_gate
        and hasattr(stage1, "velocity_gate")
        and stage1.velocity_gate is not None
    ):
        final_layer = stage1.velocity_gate[-1]
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    if args.masked_control:
        if "sparse_attention" not in payload:
            raise ValueError("masked evaluation requires sparse-control attention")
        stage1 = MaskedControlFlow(stage1).to(device)
        stage1.sparse_attention.load_state_dict(
            payload["sparse_attention"],
            strict=False,
        )
    mask_generator = None
    if args.masked_control and args.mask_policy == "learned":
        if "mask_generator" not in payload:
            raise ValueError("learned mask policy requires mask_generator checkpoint")
        mask_generator = LearnedMaskGenerator(stage1.target_channels).to(device)
        mask_generator.load_state_dict(payload["mask_generator"])
        mask_generator.eval()
    missing_token = payload.get(
        "missing_token",
        torch.zeros(stage1.target_channels),
    ).to(device)
    stage1.eval()
    distribution_head = (
        load_distribution_head(
            args.distribution_checkpoint,
            taxonomy,
            getattr(stage1, "base", stage1).output.in_channels,
            device,
        )
        if args.distribution_checkpoint is not None
        else None
    )
    room_model, structure_model, furniture_model, support_model = load_refiners(
        args,
        taxonomy,
        device,
    )
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=int(config.get("image_size", 256)),
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
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=device)
    rows = []
    for target, footprint, floor_ids, names in loader:
        target = target.to(device)
        footprint = footprint.to(device)
        floor_ids = floor_ids.to(device)
        if args.control_override_root is not None:
            controls = load_cached_contour_controls(
                args.control_override_root,
                args.split,
                names,
                device,
            )
        elif args.masked_control:
            full_source = guided_source_from_labels(
                target,
                footprint,
                taxonomy,
            )
            prepared = prepare_masked_condition(
                full_source,
                footprint,
                taxonomy,
                missing_token,
                missing_ratio=args.mask_ratio,
                floor_ids=floor_ids,
                mask_generator=mask_generator,
                mask_policy=(
                    AttributeMaskPolicy()
                    if args.mask_policy == "learned"
                    else make_mask_policy(args.mask_policy)
                ),
            )
            initial_state = None
            if distribution_head is not None:
                distribution = distribution_initial_state(
                    stage1,
                    distribution_head,
                    prepared.view.footprint,
                    prepared.view.floor_ids,
                )["state"]
                initial_state = blend_distribution_state(
                    prepared.control,
                    distribution,
                    prepared.view.footprint,
                    blend=args.distribution_blend,
                )
            sampled = sample_sparse_guided_floorplans(
                stage1,
                prepared.view.footprint,
                prepared.control,
                prepared.missing_mask,
                steps=args.flow_steps,
                floor_ids=floor_ids,
                initial_state=initial_state,
            )
            sampled = (
                prepared.control
                + float(args.cfm_residual_weight) * (sampled - prepared.control)
            )
            if args.trinity_repair:
                from floorflow.inference.trinity_refine import repair_room_raster

                room_ids = torch.tensor(taxonomy.room_ids, device=device)
                sampled_rooms = room_ids[
                    sampled[:, room_ids - 1].argmax(dim=1)
                ]
                sampled_rooms = sampled_rooms * footprint.long()
                repaired = repair_room_raster(
                    sampled_rooms,
                    footprint,
                    taxonomy,
                    steps=args.trinity_refine_steps,
                    lr=args.trinity_refine_lr,
                )
                lookup = torch.full(
                    (taxonomy.num_classes,),
                    -1,
                    device=device,
                    dtype=torch.long,
                )
                lookup[room_ids] = torch.arange(len(room_ids), device=device)
                local = lookup[repaired].clamp_min(0)
                one_hot = F.one_hot(local, len(room_ids)).permute(0, 3, 1, 2).float()
                present = torch.isin(repaired, room_ids)[:, None].float()
                sampled[:, room_ids - 1] = one_hot * present
            controls = controls_from_target_channels(
                sampled,
                footprint,
                taxonomy,
                structure_size=args.structure_size,
                furniture_size=args.furniture_size,
                support_size=args.support_size,
            )
        elif args.guided_control:
            source = guided_source_from_labels(
                target,
                footprint,
                taxonomy,
            )
            sampled = sample_guided_floorplans(
                stage1,
                footprint,
                source,
                steps=args.flow_steps,
                floor_ids=floor_ids,
            )
            sampled = (
                source
                + float(args.cfm_residual_weight) * (sampled - source)
            )
            controls = controls_from_target_channels(
                sampled,
                footprint,
                taxonomy,
                structure_size=args.structure_size,
                furniture_size=args.furniture_size,
                support_size=args.support_size,
            )
        else:
            controls = predict_contour_controls(
                stage1,
                footprint,
                taxonomy,
                floor_ids=floor_ids,
                flow_steps=args.flow_steps,
                control_source=args.control_source,
                use_room_instance_head=not args.disable_room_instance_head,
                structure_size=args.structure_size,
                furniture_size=args.furniture_size,
                support_size=args.support_size,
            )
        if args.bypass_room_refiner:
            room_logits = F.interpolate(
                controls.room_logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            room_logits = room_model(
                F.interpolate(
                    controls.room_logits,
                    size=(args.room_size, args.room_size),
                    mode="bilinear",
                    align_corners=False,
                ),
                footprint,
            )
        rooms = gate_room_labels_to_program(
            room_ids[room_logits.argmax(dim=1)],
            controls.program,
            taxonomy.room_ids,
        ) * footprint.long()
        structure = structure_model(controls.structure)
        candidates = structural_candidate_mask(
            rooms,
            footprint,
            taxonomy,
            radius=4,
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
            (
                F.interpolate(
                    structure["opening_logits"].sigmoid(),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                >= args.opening_threshold
            )
            & candidates,
            opening,
            torch.zeros_like(opening),
        )
        wall = (
            F.interpolate(
                structure["wall_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            >= args.wall_threshold
        ) & footprint.bool() & (opening == 0)
        furniture = furniture_model(
            controls.furniture,
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
            )[:, 0]
            >= args.furniture_threshold,
            furniture_labels,
            torch.zeros_like(furniture_labels),
        ) * footprint.long()
        support_probability = support_model(
            controls.support,
            output_size=target.shape[-2:],
        ).sigmoid()[:, 0]
        predicted_support = support_probability >= 0.5
        layers = LayeredFloorplan(
            room_labels=rooms,
            wall=wall.long(),
            opening=opening,
            furniture=furniture_labels,
            footprint=footprint,
            taxonomy=taxonomy,
        )
        prediction = layers.visible_labels()
        topology_wall = wall | partition_wall_mask(rooms, radius=2)
        topology_wall = F.max_pool2d(
            topology_wall.float()[:, None], 3, stride=1, padding=1
        )[:, 0] > 0.5
        closed = LayeredFloorplan(
            room_labels=rooms,
            wall=topology_wall.long(),
            opening=opening,
            furniture=furniture_labels,
            footprint=footprint,
            taxonomy=taxonomy,
        ).visible_labels()
        for index, name in enumerate(names):
            target_room = torch.isin(target[index], room_ids)
            predicted_room = torch.isin(rooms[index], room_ids)
            room_suite = room_metric_suite(
                rooms[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=predicted_room.cpu().numpy(),
                predicted_instance_support=predicted_room.cpu().numpy(),
                target_room_support=target_room.cpu().numpy(),
            )
            instance = room_suite["matched_instance"]
            channel = object_group_metrics(
                {
                    "wall": wall[index : index + 1].long(),
                    "opening": opening[index : index + 1],
                    "furniture": furniture_labels[index : index + 1],
                },
                target[index : index + 1],
                taxonomy,
            )
            fpeval = safe_fpeval(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            closed_fpeval = safe_fpeval(
                closed[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            with np.load(geometry_path(args.geometry_root, args.split, name)) as geometry:
                target_support = geometry["room_support"] > 0
            inside = footprint[index].cpu().numpy().astype(bool)
            summary = fpeval["summary"]["0.05"]
            rows.append(
                {
                    "name": name,
                    "room_macro_iou": room_suite["visible_semantic"]["room_macro_iou"],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                    **channel,
                    "fpeval_wall_f1": summary["wall_f1"],
                    "fpeval_opening_f1": summary["opening_f1"],
                    "fpeval_room_f1": summary["room_f1"],
                    "fpeval_validity": float(fpeval["prediction_validity"]["valid"]),
                    "fpeval_watertight": float(
                        closed_fpeval["prediction_validity"]["watertight"]
                    ),
                    "target_watertight": float(
                        closed_fpeval["target_validity"]["watertight"]
                    ),
                    "room_target_recall": float(
                        (predicted_room & target_room).sum() / target_room.sum().clamp_min(1)
                    ),
                    "outside_footprint_pixels": float(
                        ((prediction[index] != 0) & ~footprint[index].bool()).sum()
                    ),
                    "room_support_iou": support_iou(
                        predicted_support[index].cpu().numpy(),
                        target_support,
                        inside,
                    ),
                }
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "name"
    }
    metrics["conditional_watertight"] = float(
        np.mean(
            [
                row["fpeval_watertight"]
                for row in rows
                if row["target_watertight"] > 0.5
            ]
        )
    )
    report = {
        "cases": len(rows),
        "task_variant": "contour-only MiTA-CFM + predicted multi-channel controls",
        "flow_steps": args.flow_steps,
        "guided_control": args.guided_control,
        "masked_control": args.masked_control,
        "mask_ratio": args.mask_ratio if args.masked_control else None,
        "mask_policy": args.mask_policy if args.masked_control else None,
        "cfm_residual_weight": args.cfm_residual_weight,
        "trinity_repair": args.trinity_repair,
        "trinity_refine_steps": (
            args.trinity_refine_steps if args.trinity_repair else None
        ),
        "best_of_n_used": False,
        "seed": args.seed,
        "control_override_root": str(args.control_override_root) if args.control_override_root else None,
        "target_used_at_inference": args.control_override_root is None,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "per_case.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
