#!/usr/bin/env python3
"""Screen target-free generation plausibility before optimizing Room IoU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_raster_control_router_screen import (
    _decode_layers,
    _postprocess,
    _structure_outputs,
)
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import (
    PlausibilityThresholds,
    generation_plausibility,
)
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.geometry.core import compose_known_exterior
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.heads.raster_control_bank import load_raster_room_model
from floorflow.heads.structured_objects import partition_wall_mask
from floorflow.inference.canonical_layer_composition import (
    canonical_union_labels,
)
from floorflow.inference.room_polygon_readout import vectorize_room_masks
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
        default=ROOT / "outputs/train/raster_structure_bridge_full_v1/best.pt",
    )
    parser.add_argument(
        "--room-checkpoint",
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--room-size", type=int, default=224)
    parser.add_argument("--structure-size", type=int, default=128)
    parser.add_argument("--room-base-channels", type=int, default=24)
    parser.add_argument("--structure-base-channels", type=int, default=32)
    parser.add_argument("--boundary-width", type=int, default=7)
    parser.add_argument("--reconcile-gap-fraction", type=float, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("exterior", "repair", "union1", "union2", "partition", "vector_1.5"),
    )
    parser.add_argument("--parsable-fallback", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/predicted_control_plausibility_screen64_v2.json",
    )
    return parser.parse_args()


def _union_postprocess(
    layers: LayeredFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int,
    boundary_width: int,
) -> torch.Tensor:
    return canonical_union_labels(
        layers,
        taxonomy,
        radius=radius,
        boundary_width=boundary_width,
    )


def _partition_only_postprocess(
    layers: LayeredFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    radius: int,
    boundary_width: int,
) -> torch.Tensor:
    rooms = layers.room_labels * layers.footprint.long()
    wall = partition_wall_mask(rooms, radius=radius) & layers.footprint.bool()
    opening = torch.where(
        wall,
        layers.opening,
        torch.zeros_like(layers.opening),
    )
    layers = LayeredFloorplan(
        room_labels=layers.room_labels,
        wall=wall.long(),
        opening=opening,
        furniture=layers.furniture,
        footprint=layers.footprint,
        taxonomy=taxonomy,
    )
    return compose_known_exterior(
        layers.visible_labels(),
        layers.footprint,
        boundary_width=boundary_width,
        opening_ids=taxonomy.opening_ids,
    )


def _variant_labels(
    layers: LayeredFloorplan,
    variant: str,
    taxonomy: FloorplanTaxonomy,
    *,
    boundary_width: int,
) -> torch.Tensor:
    if variant.startswith("partition"):
        suffix = variant.removeprefix("partition")
        return _partition_only_postprocess(
            layers,
            taxonomy,
            radius=int(suffix) if suffix else 1,
            boundary_width=boundary_width,
        )
    if variant.startswith("union"):
        return _union_postprocess(
            layers,
            taxonomy,
            radius=int(variant[-1]),
            boundary_width=boundary_width,
        )
    if variant == "exterior":
        return _postprocess(layers, "exterior", taxonomy)
    return _postprocess(layers, variant, taxonomy)


def _with_parsable_fallback(
    labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> tuple[torch.Tensor, bool]:
    label_np = np.squeeze(labels[0].cpu().numpy())
    footprint_np = np.squeeze(footprint.cpu().numpy())
    if footprint_np.size == label_np.size:
        footprint_np = footprint_np.reshape(label_np.shape)
    metrics = generation_plausibility(
        label_np,
        footprint_np,
        taxonomy,
    )
    if metrics["parsable"] == 1.0:
        return labels, False
    fallback = vectorize_room_masks(
        label_np,
        footprint_np.astype(bool),
        taxonomy,
        epsilon=1.5,
    )
    return torch.from_numpy(fallback)[None].to(labels.device), True


def _room_iou_diagnostic(
    labels: np.ndarray,
    target: np.ndarray,
    footprint: np.ndarray,
    taxonomy: FloorplanTaxonomy,
) -> float:
    suite = room_metric_suite(
        labels,
        target,
        footprint,
        taxonomy,
        predicted_room_support=np.isin(labels, taxonomy.room_ids),
        predicted_instance_support=np.isin(labels, taxonomy.room_ids),
        target_room_support=np.isin(target, taxonomy.room_ids),
    )
    return float(suite["visible_semantic"]["room_macro_iou"])


def _summarize(rows: list[dict]) -> dict:
    numeric = {}
    for key in rows[0]:
        if key in {"case", "raster_failure_codes", "plan_failure_codes"}:
            continue
        values = [row[key] for row in rows]
        if all(
            isinstance(value, (bool, np.bool_))
            for value in values
        ) or all(
            isinstance(value, (int, float, np.integer, np.floating))
            for value in values
        ):
            numeric[key] = float(np.mean(values))
    raster_failures = Counter(
        code for row in rows for code in row["raster_failure_codes"]
    )
    plan_failures = Counter(
        code for row in rows for code in row["plan_failure_codes"]
    )
    return {
        "cases": len(rows),
        "metrics": numeric,
        "raster_failure_counts": dict(sorted(raster_failures.items())),
        "plan_failure_counts": dict(sorted(plan_failures.items())),
    }


@torch.inference_mode()
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
        persistent_workers=args.workers > 0,
    )
    room_model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    structure_model, _structure_config = load_joint_control_repair(
        args.structure_checkpoint,
        taxonomy,
        device,
    )
    room_model.eval()
    structure_model.eval()
    rows = {variant: [] for variant in args.variants}
    fallback_counts = Counter()
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
        room_logits = F.interpolate(
            room_model(footprint, control_image=image),
            size=(args.room_size, args.room_size),
            mode="bilinear",
            align_corners=False,
        )
        output = _structure_outputs(
            structure_model,
            predicted,
            room_logits,
            footprint,
            args,
            repair_room=False,
        )
        layers, _rooms, _wall, _opening = _decode_layers(
            room_logits,
            output["structure"],
            footprint,
            taxonomy,
        )
        for variant in args.variants:
            label_variant = variant.removesuffix("_explicit")
            explicit_rooms = variant.endswith("_explicit")
            labels = _variant_labels(
                layers,
                label_variant,
                taxonomy,
                boundary_width=args.boundary_width,
            )
            for index, name in enumerate(names):
                case_labels = labels[index]
                fallback_used = False
                if args.parsable_fallback:
                    case_labels, fallback_used = _with_parsable_fallback(
                        case_labels[None],
                        footprint[index],
                        taxonomy,
                    )
                    case_labels = case_labels[0]
                    fallback_counts[variant] += int(fallback_used)
                labels_np = case_labels.cpu().numpy()
                target_np = target[index].cpu().numpy()
                footprint_np = footprint[index].cpu().numpy()
                footprint_np = np.squeeze(footprint_np)
                if footprint_np.size == labels_np.size:
                    footprint_np = footprint_np.reshape(labels_np.shape)
                rows[variant].append(
                    {
                        "case": str(name),
                        **generation_plausibility(
                            labels_np,
                            footprint_np,
                            taxonomy,
                            boundary_width=args.boundary_width,
                            explicit_rooms=explicit_rooms,
                            reconcile_gap_fraction=args.reconcile_gap_fraction,
                        ),
                        "room_macro_iou_diagnostic": _room_iou_diagnostic(
                            labels_np,
                            target_np,
                            footprint_np,
                            taxonomy,
                        ),
                        "fallback_used": fallback_used,
                    }
                )
    report = {
        "schema_version": "predicted-control-plausibility-screen-v2",
        "scope": "mechanism screening only; Room IoU is diagnostic and non-gating",
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "target_used_for_diagnostic_only": True,
        "thresholds": PlausibilityThresholds().__dict__,
        "fallback_counts": dict(fallback_counts),
        "arms": {
            variant: _summarize(case_rows)
            for variant, case_rows in rows.items()
        },
        "cases": {
            variant: case_rows
            for variant, case_rows in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {variant: result["metrics"] for variant, result in report["arms"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
