#!/usr/bin/env python3
"""Evaluate the complete layered floorplan generation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.data.control_cache import (
    CachedControlDataset,
    CachedValidityDataset,
    resize_control,
)
from floorflow.data.control_contract import oracle_control_payload
from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.evaluation.completion_utility import completion_utility_metrics
from floorflow.evaluation.control_baseline import control_layers, control_layers_batch
from floorflow.evaluation.fp_completion_pipeline import (
    geometry_path,
    load_completion_modules,
    load_control_manifest,
    support_iou,
)
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.coarse_furniture_refiner import coarse_furniture_control
from floorflow.heads.coarse_structure_refiner import coarse_structure_control
from floorflow.heads.coarse_support_refiner import coarse_support_control
from floorflow.heads.geometry_heads import structural_candidate_mask
from floorflow.heads.room_program import (
    gate_room_labels_to_program,
    room_presence_target,
)
from floorflow.heads.structured_objects import (
    object_group_metrics,
    partition_wall_mask,
)
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.representation.layered_floorplan import LayeredFloorplan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-checkpoint", type=Path, required=True)
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--layout-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--furniture-size", type=int, default=192)
    parser.add_argument("--support-size", type=int, default=192)
    parser.add_argument("--wall-threshold", type=float, default=0.6)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--furniture-threshold", type=float, default=0.5)
    parser.add_argument(
        "--room-control-blend",
        type=float,
        default=0.0,
        help="Add this weight times the supplied coarse room logits before argmax.",
    )
    parser.add_argument(
        "--room-mode-kernel",
        type=int,
        default=1,
        help="Odd local-mode kernel for topology-aware room readout.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-components",
        nargs="+",
        choices=("room", "structure", "furniture", "support"),
        default=(),
        help="Use the matched raw-control path for selected stages.",
    )
    parser.add_argument(
        "--preserve-room-controls",
        action="store_true",
        help="Keep supplied room logits outside serialized missing regions.",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=None,
        help="Read serialized FP control bundles instead of deriving them from target.",
    )
    parser.add_argument(
        "--control-manifest",
        type=Path,
        default=None,
        help="Manifest path; defaults to <control-root>/<split>/manifest.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        cache_root=args.cache_root,
        limit=args.limit,
        augment=False,
    )
    taxonomy = dataset.taxonomy
    control_manifest = load_control_manifest(
        control_root=args.control_root,
        control_manifest=args.control_manifest,
        split=args.split,
        dataset=dataset,
        limit=args.limit,
    )
    if args.control_root is not None:
        dataset_class = (
            CachedValidityDataset
            if args.preserve_room_controls
            else CachedControlDataset
        )
        dataset = dataset_class(dataset, args.control_root, args.split)
    room_model, structure_model, furniture_model, support_model = load_completion_modules(
        args.room_checkpoint,
        args.structure_checkpoint,
        args.furniture_checkpoint,
        args.support_checkpoint,
        taxonomy,
        device,
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
    rows: list[dict[str, float]] = []
    raster_dir = args.output_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        masks = None
        if len(batch) == 6:
            target, footprint, _floor_ids, names, controls, masks = batch
            controls = {
                key: value.to(device) for key, value in controls.items()
            }
        elif len(batch) == 5:
            target, footprint, _floor_ids, names, controls = batch
            controls = {
                key: value.to(device) for key, value in controls.items()
            }
        else:
            target, footprint, _floor_ids, names = batch
            controls = None
        target = target.to(device)
        footprint = footprint.to(device)
        control_bundle = controls
        if control_bundle is None:
            control_bundle = oracle_control_payload(
                target,
                footprint,
                taxonomy,
                room_size=args.layout_size,
                structure_size=args.structure_size,
                furniture_size=args.furniture_size,
                support_size=args.support_size,
            )
        room_control = (
            resize_control(controls["room_logits"], args.layout_size)
            if controls is not None
            else coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.layout_size,
            )
        )
        room_logits = room_model(
            room_control,
            footprint,
        )
        if controls is not None and args.room_control_blend > 0.0:
            room_logits = room_logits + args.room_control_blend * F.interpolate(
                controls["room_logits"],
                size=room_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if masks is not None and args.preserve_room_controls:
            valid_room = masks["valid_room_logits"].to(device).bool()
            supplied_room = F.interpolate(
                controls["room_logits"],
                size=room_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            valid_room = F.interpolate(
                valid_room[:, None].float(),
                size=room_logits.shape[-2:],
                mode="nearest",
            ).bool()
            room_logits = torch.where(
                valid_room,
                supplied_room,
                room_logits,
            )
        program = (
            controls["program"].bool()
            if controls is not None
            else room_presence_target(target, taxonomy)
        )
        room_labels = gate_room_labels_to_program(
            room_ids[room_logits.argmax(dim=1)],
            program,
            taxonomy.room_ids,
        ) * footprint.long()
        if args.room_mode_kernel > 1:
            kernel = int(args.room_mode_kernel)
            if kernel % 2 == 0:
                raise ValueError("room mode kernel must be odd")
            padded = F.pad(
                room_labels.float()[:, None],
                (kernel // 2,) * 4,
                mode="replicate",
            )
            patches = F.unfold(padded, kernel).reshape(
                room_labels.shape[0],
                kernel * kernel,
                *room_labels.shape[-2:],
            )
            room_labels = patches.mode(dim=1).values.long() * footprint.long()

        structure = structure_model(
            resize_control(controls["structure"], args.structure_size)
            if controls is not None
            else coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
        )
        candidate = structural_candidate_mask(
            room_labels,
            footprint,
            taxonomy,
            radius=4,
        )
        opening_labels = opening_ids[
            F.interpolate(
                structure["opening_type_logits"],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
        ]
        opening_labels = torch.where(
            (
                F.interpolate(
                    structure["opening_logits"].sigmoid(),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                >= args.opening_threshold
            )
            & candidate,
            opening_labels,
            torch.zeros_like(opening_labels),
        )
        wall = (
            F.interpolate(
                structure["wall_logits"].sigmoid(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            >= args.wall_threshold
        ) & footprint.bool() & (opening_labels == 0)

        furniture = furniture_model(
            (
                resize_control(controls["furniture"], args.furniture_size)
                if controls is not None
                else coarse_furniture_control(
                    target,
                    footprint,
                    taxonomy,
                    size=args.furniture_size,
                )
            ),
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
            (
                resize_control(controls["support"], args.support_size)
                if controls is not None
                else coarse_support_control(
                    target,
                    footprint,
                    taxonomy,
                    size=args.support_size,
                )
            ),
            output_size=target.shape[-2:],
        ).sigmoid()[:, 0]
        predicted_support = support_probability >= 0.5

        if args.disable_components:
            raw_rooms, raw_wall, raw_openings, raw_furniture = control_layers_batch(
                control_bundle,
                footprint,
                taxonomy,
            )
            if "room" in args.disable_components:
                room_labels = raw_rooms
            if "structure" in args.disable_components:
                wall = raw_wall.bool()
                opening_labels = raw_openings
            if "furniture" in args.disable_components:
                furniture_labels = raw_furniture
            if "support" in args.disable_components:
                predicted_support = torch.isin(room_labels, room_ids)

        layers = LayeredFloorplan(
            room_labels=room_labels,
            wall=wall.long(),
            opening=opening_labels,
            furniture=furniture_labels,
            footprint=footprint,
            taxonomy=taxonomy,
        )
        prediction = layers.visible_labels()
        topology_wall = wall | partition_wall_mask(room_labels, radius=2)
        topology_wall = F.max_pool2d(
            topology_wall.float()[:, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[:, 0] > 0.5
        closed_layers = LayeredFloorplan(
            room_labels=room_labels,
            wall=topology_wall.long(),
            opening=opening_labels,
            furniture=furniture_labels,
            footprint=footprint,
            taxonomy=taxonomy,
        )
        closed_prediction = closed_layers.visible_labels()

        for index, name in enumerate(names):
            raw_rooms, raw_wall, raw_openings, raw_furniture = control_layers(
                {
                    key: value[index]
                    for key, value in control_bundle.items()
                },
                footprint[index],
                taxonomy,
            )
            raw_prediction = LayeredFloorplan(
                room_labels=raw_rooms[None],
                wall=raw_wall[None],
                opening=raw_openings[None],
                furniture=raw_furniture[None],
                footprint=footprint[index : index + 1],
                taxonomy=taxonomy,
            ).visible_labels()[0]
            room_suite = room_metric_suite(
                room_labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(
                    room_labels[index],
                    room_ids,
                ).cpu().numpy(),
                predicted_instance_support=torch.isin(
                    room_labels[index],
                    room_ids,
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = room_suite["matched_instance"]
            channel = object_group_metrics(
                {
                    "wall": wall[index : index + 1].long(),
                    "opening": opening_labels[index : index + 1],
                    "furniture": furniture_labels[index : index + 1],
                },
                target[index : index + 1],
                taxonomy,
            )
            fpeval = fpeval_raster_metrics(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            closed_fpeval = fpeval_raster_metrics(
                closed_prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            room_fpeval = fpeval_raster_metrics(
                prediction[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
                explicit_prediction_rooms=True,
                explicit_target_rooms=True,
            )
            summary = fpeval["summary"]["0.05"]
            path = geometry_path(args.geometry_root, args.split, name)
            with np.load(path) as geometry:
                target_support = geometry["room_support"] > 0
                occlusion = geometry["occlusion_mask"] > 0
            support_mask = predicted_support[index].cpu().numpy()
            room_label_support = torch.isin(
                room_labels[index],
                room_ids,
            ).cpu().numpy()
            inside = footprint[index].cpu().numpy().astype(bool)
            metrics = {
                "room_macro_iou": room_suite["visible_semantic"]["room_macro_iou"],
                "instance_geometric_iou": instance["matched_geometric_iou"],
                "instance_room_f1": instance["instance_f1_at_threshold"],
                "instance_type_accuracy": instance["matched_type_accuracy"],
                **channel,
                "fpeval_wall_f1": summary["wall_f1"],
                "fpeval_opening_f1": summary["opening_f1"],
                "fpeval_room_polygon_f1": room_fpeval["summary"]["0.05"][
                    "room_f1"
                ],
                "fpeval_room_face_f1": summary["room_f1"],
                "fpeval_validity": float(
                    fpeval["prediction_validity"]["valid"]
                ),
                "fpeval_watertight": float(
                    closed_fpeval["prediction_validity"]["watertight"]
                ),
                "target_watertight": float(
                    closed_fpeval["target_validity"]["watertight"]
                ),
                "room_support_iou": support_iou(
                    support_mask,
                    target_support,
                    inside,
                ),
                "room_support_iou_visible": support_iou(
                    support_mask,
                    target_support,
                    inside & ~occlusion,
                ),
                "room_label_support_iou": support_iou(
                    room_label_support,
                    target_support,
                    inside,
                ),
                **completion_utility_metrics(
                    prediction[index].cpu().numpy(),
                    raw_prediction.cpu().numpy(),
                    target[index].cpu().numpy(),
                    taxonomy,
                    output_edit_cost=summary["edit_cost"],
                    output_edit_cost_per_gt_wall=summary[
                        "edit_cost_per_gt_wall"
                    ],
                ),
            }
            rows.append(metrics)
            np.savez_compressed(
                raster_dir / f"{name.replace('/', '__')}.npz",
                prediction=prediction[index].cpu().numpy().astype(np.uint8),
                topology_prediction=closed_prediction[index].cpu().numpy().astype(np.uint8),
                target=target[index].cpu().numpy().astype(np.uint8),
                footprint=footprint[index].cpu().numpy().astype(np.uint8),
                room_labels=room_labels[index].cpu().numpy().astype(np.uint8),
                room_support=predicted_support[index].cpu().numpy().astype(np.uint8),
            )
        print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)

    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
    target_watertight = [
        row["target_watertight"] > 0.5
        for row in rows
    ]
    metrics["conditional_watertight"] = float(
        np.mean(
            [
                row["fpeval_watertight"]
                for row, valid in zip(rows, target_watertight)
                if valid
            ]
        )
    )
    report = {
        "cases": len(rows),
        "task_variant": (
            "serialized control contract + layered CAD completion"
            if control_manifest is not None
            else "contour + room program + coarse room/structure/furniture layout"
        ),
        "best_of_n_used": False,
        "single_deterministic_sample": True,
        "disabled_components": sorted(args.disable_components),
        "control_contract": (
            {
                "source": control_manifest["source"],
                "target_used_for_control": control_manifest[
                    "target_used_for_control"
                ],
                "deployable": control_manifest["deployable"],
                "root": control_manifest["output_root"],
                "manifest": control_manifest["manifest_path"],
                "manifest_sha256": control_manifest["manifest_sha256"],
                "taxonomy_sha256": control_manifest.get("taxonomy_sha256"),
            }
            if control_manifest is not None
            else None
        ),
        "metrics": metrics,
        "thresholds": {
            "wall": args.wall_threshold,
            "opening": args.opening_threshold,
            "furniture": args.furniture_threshold,
        },
        "target_watertight_fraction": float(np.mean(target_watertight)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "per_case.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
