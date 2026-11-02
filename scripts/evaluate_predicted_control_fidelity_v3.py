#!/usr/bin/env python3
"""Evaluate predicted-control fidelity on the canonical explicit-room plan."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_predicted_control_plausibility import _union_postprocess
from evaluate_raster_control_router_screen import (
    _decode_layers,
    _structure_outputs,
)
from predicted_control_fidelity_v3_args import parse_args
from predicted_control_fidelity_v3_report import report_config
from train_raster_predicted_control_bridge import RasterPredictedControlDataset
from train_residential_predicted_refiner import _predicted_controls

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.generation_plausibility import generation_plausibility
from floorflow.evaluation.predicted_control_fidelity import (
    predicted_control_fidelity_metrics,
)
from floorflow.evaluation.predicted_control_reporting import (
    numeric_row_summary,
    opening_metric_kwargs,
    opening_runtime_tensors,
)
from floorflow.geometry.wall_graph_readout import (
    WallSkeletonPolicy,
    raster_labels_to_wall_plan,
)
from floorflow.heads.joint_control_loading import load_joint_control_repair
from floorflow.heads.raster_control_bank import (
    RasterInstanceClassRoutedAdapter,
    RasterInstanceRoomAdapter,
    RasterInstanceSoftRoutedAdapter,
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.heads.room_mask_fusion import mask_region_class_logits
from floorflow.inference.opening_candidate_batch import (
    rank_center_width_openings,
)
from floorflow.inference.opening_candidate_ranker import (
    load_opening_candidate_ranker,
)
from floorflow.inference.opening_heatmap_decoder import (
    load_opening_heatmap_decoder,
)
from floorflow.inference.room_identity_arbitration import (
    arbitrate_room_logits,
    load_room_identity_arbitration_gate,
    room_identity_arbitration_features,
)
from floorflow.inference.room_repair import merge_room_fragments
from floorflow.inference.structure_evidence import (
    apply_structure_evidence,
    load_geometry_wall_model,
    load_structured_opening_model,
)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    wall_readout_policy = WallSkeletonPolicy(
        close_kernel=int(args.wall_skeleton_close_kernel),
        simplification_fraction=float(
            args.wall_skeleton_simplification_fraction
        ),
        minimum_segment_length=float(
            args.wall_skeleton_min_segment_length
        ),
    )
    wall_readout_policy.validate()
    taxonomy = FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    dataset = RasterPredictedControlDataset(
        args.predicted_root,
        "val",
        args.cache_root,
        args.data_root,
        image_size=args.image_size,
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
    arbitration_gate = None
    query_model = None
    gate_threshold = args.room_identity_gate_threshold
    if args.room_identity_gate is not None:
        arbitration_gate = load_room_identity_arbitration_gate(
            args.room_identity_gate,
            device,
        )
        gate_payload = torch.load(
            args.room_identity_gate,
            map_location="cpu",
            weights_only=False,
        )
        if gate_threshold is None:
            gate_threshold = float(gate_payload.get("config", {}).get("threshold", 0.5))
        query_model = load_raster_instance_control_model(
            args.room_query_checkpoint,
            taxonomy,
            device,
        )
        room_model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    elif args.room_instance_blend is None:
        room_model = load_raster_room_model(args.room_checkpoint, taxonomy, device)
    else:
        instance_model = load_raster_instance_control_model(
            args.room_checkpoint,
            taxonomy,
            device,
        )
        if args.room_instance_route == "class_region":
            room_model = RasterInstanceClassRoutedAdapter(
                instance_model,
                threshold=args.room_mask_threshold,
                strength=args.room_mask_strength,
            ).to(device)
        elif args.room_instance_route == "soft":
            room_model = RasterInstanceSoftRoutedAdapter(
                instance_model,
                threshold=args.room_mask_threshold,
                strength=args.room_mask_strength,
            ).to(device)
        else:
            room_model = RasterInstanceRoomAdapter(
                instance_model,
                args.room_instance_blend,
            ).to(device)
    structure_model, structure_config = load_joint_control_repair(
        args.structure_checkpoint,
        taxonomy,
        device,
    )
    geometry_model = None
    if args.wall_centerline_checkpoint is not None:
        geometry_model = load_geometry_wall_model(
            args.wall_centerline_checkpoint,
            device,
            taxonomy,
        )
    object_backbone = None
    object_head = None
    if args.structured_object_checkpoint is not None:
        object_backbone, object_head = load_structured_opening_model(
            args.structured_object_checkpoint,
            device,
            taxonomy,
        )
    opening_runtime = (
        load_opening_heatmap_decoder(
            args.opening_heatmap_checkpoint,
            taxonomy,
            device,
        )
        if args.opening_heatmap_checkpoint is not None
        else None
    )
    opening_candidate_ranker = (
        load_opening_candidate_ranker(
            args.opening_heatmap_candidate_ranker,
            device=device,
        )
        if args.opening_heatmap_candidate_ranker is not None
        else None
    )
    if (
        opening_candidate_ranker is not None
        and args.opening_heatmap_candidate_ranker_threshold is not None
    ):
        opening_candidate_ranker = replace(
            opening_candidate_ranker,
            presence_threshold=float(
                args.opening_heatmap_candidate_ranker_threshold
            ),
        )
    room_model.eval()
    structure_model.eval()
    rows = []
    for target, footprint, floor_ids, names, image in loader:
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
        arbitration_selected = None
        dense_structure = None
        dense_logits = None
        prediction_plans = None
        if arbitration_gate is not None:
            dense_logits = F.interpolate(
                room_model(footprint, control_image=image),
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            query_output = query_model(footprint, control_image=image)
            query_logits = F.interpolate(
                query_output["fused_logits"],
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            room_footprint = F.interpolate(
                footprint[:, None].float(),
                size=(args.room_size, args.room_size),
                mode="nearest",
            )[:, 0].bool()
            routed_logits = mask_region_class_logits(
                {
                    "mask_logits": F.interpolate(
                        query_output["mask_logits"],
                        size=(args.room_size, args.room_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                },
                dense_logits,
                room_footprint,
                threshold=args.room_mask_threshold,
                strength=args.room_mask_strength,
            )
            features = room_identity_arbitration_features(
                dense_logits,
                query_logits,
                routed_logits,
                room_footprint,
                tuple(taxonomy.room_ids),
            )
            room_logits, arbitration_selected = arbitrate_room_logits(
                arbitration_gate,
                dense_logits,
                routed_logits,
                features,
                threshold=float(gate_threshold),
            )
            structure_room_logits = (
                dense_logits
                if args.structure_room_source == "dense"
                or args.room_identity_safety_fallback
                else room_logits
            )
        elif (
            (
                args.structure_room_source == "dense"
                or args.room_identity_safety_fallback
            )
            and hasattr(room_model, "forward_with_dense")
        ):
            room_model_output, structure_room_output = room_model.forward_with_dense(
                footprint,
                control_image=image,
            )
        else:
            room_model_output = room_model(footprint, control_image=image)
            structure_room_output = room_model_output
        if arbitration_gate is None:
            room_logits = F.interpolate(
                room_model_output,
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            structure_room_logits = F.interpolate(
                structure_room_output,
                size=(args.room_size, args.room_size),
                mode="bilinear",
                align_corners=False,
            )
            if (
                args.room_identity_safety_fallback
                and args.room_instance_route == "class_region"
                and args.room_instance_blend is not None
            ):
                dense_logits = structure_room_logits
                arbitration_selected = torch.ones(
                    room_logits.shape[0],
                    dtype=torch.bool,
                    device=device,
                )
        if args.room_identity_safety_fallback and dense_logits is not None:
            structure_room_logits = dense_logits
            dense_structure = _structure_outputs(
                structure_model,
                predicted,
                dense_logits,
                footprint,
                args,
                repair_room=False,
            )["structure"]
        output = (
            {"structure": dense_structure}
            if dense_structure is not None
            else _structure_outputs(
                structure_model,
                predicted,
                structure_room_logits,
                footprint,
                args,
                repair_room=False,
            )
        )
        layers, _rooms, _wall, _opening = _decode_layers(
            room_logits,
            output["structure"],
            footprint,
            taxonomy,
        )
        labels = _union_postprocess(
            layers,
            taxonomy,
            radius=0,
            boundary_width=args.boundary_width,
        )
        opening_host_plans = [
            raster_labels_to_wall_plan(
                row,
                taxonomy,
                wall_policy=wall_readout_policy,
            )
            for row in labels.cpu().numpy()
        ]
        labels, prediction_plans = apply_structure_evidence(
            labels,
            _rooms,
            target,
            footprint,
            floor_ids.to(device),
            taxonomy,
            geometry_model=geometry_model,
            object_backbone=object_backbone,
            object_head=object_head,
            wall_threshold=args.wall_centerline_threshold,
            wall_dilation=args.wall_centerline_dilation,
            opening_protect_radius=args.wall_centerline_opening_protect_radius,
            wall_mode=args.wall_centerline_mode,
            opening_threshold=args.structured_opening_threshold,
        )
        safety_fallback = torch.zeros(
            labels.shape[0],
            dtype=torch.bool,
            device=device,
        )
        if (
            args.room_identity_safety_fallback
            and dense_logits is not None
            and dense_structure is not None
            and arbitration_selected is not None
        ):
            dense_layers, _rooms, _wall, _opening = _decode_layers(
                dense_logits,
                dense_structure,
                footprint,
                taxonomy,
            )
            dense_labels = _union_postprocess(
                dense_layers,
                taxonomy,
                radius=0,
                boundary_width=args.boundary_width,
            )
            for index in range(labels.shape[0]):
                plausibility = generation_plausibility(
                    labels[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    explicit_rooms=True,
                )
                if arbitration_selected[index] and (
                    not bool(plausibility["plan_plausible"])
                    or float(plausibility["wall_continuity_rate"])
                    < args.room_safety_min_wall_continuity
                ):
                    labels[index] = dense_labels[index]
                    safety_fallback[index] = True
            arbitration_selected = arbitration_selected & ~safety_fallback
        if args.room_merge_min_area > 0:
            labels = torch.stack(
                [
                    torch.from_numpy(
                        merge_room_fragments(
                            row.cpu().numpy(),
                            taxonomy,
                            min_area=args.room_merge_min_area,
                        )
                    )
                    for row in labels
                ]
            ).to(device)
        (
            opening_probability,
            opening_foreground,
            opening_family_logits,
            opening_center,
            opening_interval_width,
        ) = (
            opening_runtime_tensors(args, opening_runtime, footprint, image, taxonomy)
        )
        ranked_detections = None
        if (
            opening_candidate_ranker is not None
            and args.opening_ranker_batch_mode == "batch"
            and opening_center is not None
            and opening_interval_width is not None
            and opening_family_logits is not None
        ):
            ranked_detections = rank_center_width_openings(
                opening_host_plans,
                [value.numpy() for value in opening_center],
                [value.numpy() for value in opening_foreground],
                [value.numpy() for value in opening_interval_width],
                [value.numpy() for value in opening_family_logits],
                taxonomy,
                opening_candidate_ranker,
                threshold=args.opening_heatmap_threshold,
                min_area=args.opening_heatmap_min_area,
                min_peak_distance=args.opening_heatmap_min_peak_distance,
                frame=float(args.image_size),
            )
        for index, name in enumerate(names):
            opening_kwargs = opening_metric_kwargs(
                args,
                index,
                output["structure"],
                opening_probability,
                prediction_plans,
                opening_host_plans,
                opening_foreground,
                opening_family_logits,
                opening_candidate_ranker,
                opening_center,
                opening_interval_width,
                ranked_detections,
            )
            if ranked_detections is not None:
                opening_kwargs.update(
                    prediction_opening_candidate_ranker=None,
                    prediction_opening_precomputed_detections=(
                        ranked_detections[index]
                    ),
                )
            row = {
                "case": str(name),
                **predicted_control_fidelity_metrics(
                    labels[index].cpu().numpy(),
                    target[index].cpu().numpy(),
                    footprint[index].cpu().numpy(),
                    taxonomy,
                    reconcile_gap_fraction=args.reconcile_gap_fraction,
                    graph_ged_timeout_seconds=args.ged_timeout_seconds,
                    include_room_decomposition=args.include_room_decomposition,
                    wall_readout_policy=wall_readout_policy,
                    **opening_kwargs,
                ),
            }
            if arbitration_selected is not None:
                row["arbitration_selected"] = float(arbitration_selected[index])
                row["arbitration_safety_fallback"] = float(safety_fallback[index])
            rows.append(row)
    report = {
        "schema_version": "predicted-control-fidelity-v3",
        "scope": "canonical explicit-room target-comparative evaluation",
        "split": "val",
        "validation_cases": len(dataset),
        "validation_offset": args.offset,
        "test_opened": False,
        "target_used_at_inference": False,
        "config": report_config(
            args,
            gate_threshold,
            structure_config,
        ),
        "summary": numeric_row_summary(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
