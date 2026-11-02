#!/usr/bin/env python3
"""Evaluate the layered room/structure pipeline on skv4 test plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.skv4 import rasterize_skv4_plan
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.coarse_layout_refiner import CoarseLayoutRefiner
from floorflow.heads.coarse_structure_refiner import (
    CoarseStructureRefiner,
    coarse_structure_control,
)
from floorflow.heads.geometry_heads import structural_candidate_mask
from floorflow.heads.room_program import (
    gate_room_labels_to_program,
    room_presence_target,
)
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.inference.structure_fusion import opening_labels_from_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-checkpoint", type=Path, required=True)
    parser.add_argument("--structure-checkpoint", type=Path, required=True)
    parser.add_argument("--skv4-root", type=Path, default=Path("outputs/cache/skv4"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/cache/cubicasa5k_full_v2"),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--layout-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--wall-threshold", type=float, default=0.6)
    parser.add_argument("--opening-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-rasters", action="store_true")
    parser.add_argument("--opening-control-prior", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_models(args, taxonomy, device):
    room_payload = torch.load(
        args.room_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    room = CoarseLayoutRefiner(len(taxonomy.room_ids)).to(device)
    room.load_state_dict(room_payload["model"])
    room.eval()
    structure_payload = torch.load(
        args.structure_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    structure = CoarseStructureRefiner(
        len(taxonomy.room_ids) + len(taxonomy.opening_ids) + 3,
        len(taxonomy.opening_ids),
        base_channels=int(structure_payload["config"].get("base_channels", 32)),
    ).to(device)
    structure.load_state_dict(structure_payload["model"])
    structure.eval()
    return room, structure


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    room_model, structure_model = load_models(args, taxonomy, device)
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    opening_ids = torch.tensor(taxonomy.opening_ids, device=device)
    rows = []
    raster_dir = args.output.parent / "skv4_rasters"
    if args.save_rasters:
        raster_dir.mkdir(parents=True, exist_ok=True)
    with (args.skv4_root / "test.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if args.limit is not None and len(rows) >= args.limit:
                break
            plan = json.loads(line)
            target_np, footprint_np = rasterize_skv4_plan(
                plan,
                taxonomy,
                args.image_size,
            )
            target = torch.from_numpy(target_np).long()[None].to(device)
            footprint = torch.from_numpy(footprint_np).float()[None].to(device)
            room_logits = room_model(
                coarse_room_layout_logits(
                    target,
                    taxonomy,
                    size=args.layout_size,
                ),
                footprint,
            )
            rooms = gate_room_labels_to_program(
                room_ids[room_logits.argmax(dim=1)],
                room_presence_target(target, taxonomy),
                taxonomy.room_ids,
            ) * footprint.long()
            structure_control = coarse_structure_control(
                target,
                footprint,
                taxonomy,
                size=args.structure_size,
            )
            structure = structure_model(
                structure_control,
                output_size=target.shape[-2:],
            )
            candidate = structural_candidate_mask(
                rooms,
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
            if args.opening_control_prior:
                opening_labels = opening_labels_from_control(
                    structure_control,
                    taxonomy,
                    output_size=target.shape[-2:],
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
            layers = LayeredFloorplan(
                room_labels=rooms,
                wall=wall.long(),
                opening=opening_labels,
                furniture=torch.zeros_like(target),
                footprint=footprint,
                taxonomy=taxonomy,
            )
            prediction = layers.visible_labels()
            topology_wall = wall | partition_wall_mask(rooms, radius=2)
            topology_wall = F.max_pool2d(
                topology_wall.float()[:, None],
                3,
                stride=1,
                padding=1,
            )[:, 0] > 0.5
            closed_prediction = LayeredFloorplan(
                room_labels=rooms,
                wall=topology_wall.long(),
                opening=opening_labels,
                furniture=torch.zeros_like(target),
                footprint=footprint,
                taxonomy=taxonomy,
            ).visible_labels()
            room_suite = room_metric_suite(
                rooms[0].cpu().numpy(),
                target[0].cpu().numpy(),
                footprint[0].cpu().numpy(),
                taxonomy,
                predicted_room_support=torch.isin(
                    rooms[0],
                    room_ids,
                ).cpu().numpy(),
                predicted_instance_support=torch.isin(
                    rooms[0],
                    room_ids,
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[0],
                    room_ids,
                ).cpu().numpy(),
            )
            instance = room_suite["matched_instance"]
            fpeval = fpeval_raster_metrics(
                prediction[0].cpu().numpy(),
                target[0].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            closed_fpeval = fpeval_raster_metrics(
                closed_prediction[0].cpu().numpy(),
                target[0].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
            )
            summary = fpeval["summary"]["0.05"]
            rows.append(
                {
                    "id": plan["id"],
                    "room_macro_iou": room_suite["visible_semantic"][
                        "room_macro_iou"
                    ],
                    "instance_geometric_iou": instance["matched_geometric_iou"],
                    "instance_room_f1": instance["instance_f1_at_threshold"],
                    "instance_type_accuracy": instance["matched_type_accuracy"],
                    "target_room_pixels": int(
                        torch.isin(target, room_ids).sum().item()
                    ),
                    "fpeval_wall_f1": summary["wall_f1"],
                    "fpeval_opening_f1": summary["opening_f1"],
                    "fpeval_validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "fpeval_watertight": float(
                        closed_fpeval["prediction_validity"]["watertight"]
                    ),
                    "target_watertight": float(
                        closed_fpeval["target_validity"]["watertight"]
                    ),
                }
            )
            if args.save_rasters:
                np.savez_compressed(
                    raster_dir / f"{str(plan['id']).replace('/', '__')}.npz",
                    prediction=prediction[0].cpu().numpy().astype(np.uint8),
                    target=target[0].cpu().numpy().astype(np.uint8),
                    footprint=footprint[0].cpu().numpy().astype(np.uint8),
                    room_labels=rooms[0].cpu().numpy().astype(np.uint8),
                )
    room_keys = {
        "room_macro_iou",
        "instance_geometric_iou",
        "instance_room_f1",
        "instance_type_accuracy",
    }
    eligible = [
        row for row in rows if row["target_room_pixels"] > 0
    ]
    keys = tuple(
        key for key in rows[0]
        if key not in {"id", "target_room_pixels"}
    )
    report = {
        "dataset": "skv4",
        "split": "official_plan_test",
        "case_count": len(rows),
        "task_variant": (
            "contour + room program + coarse room/structure layout control"
        ),
        "best_of_n_used": False,
        "single_deterministic_sample": True,
        "summary": {
            key: float(
                np.mean(
                    [
                        row[key]
                        for row in (eligible if key in room_keys else rows)
                    ]
                )
            )
            for key in keys
        },
        "rows": rows,
        "room_metric_cases": len(eligible),
        "no_room_gt_cases": len(rows) - len(eligible),
    }
    report["summary"]["conditional_watertight"] = float(
        np.mean(
            [
                row["fpeval_watertight"]
                for row in rows
                if row["target_watertight"] > 0.5
            ]
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
