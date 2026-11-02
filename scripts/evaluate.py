#!/usr/bin/env python3
"""Evaluate a trained floorplan flow model over an entire CubiCasa split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import CubiCasaFloorplanDataset
from floorflow.training.checkpoint import load_checkpoint_model
from floorflow.representation.contour_memory import ContourPartitionMemory
from floorflow.objectives.condition_objective import room_macro_iou_from_labels
from floorflow.flow.core import decode_floorplans, sample_floorplans
from floorflow.geometry.core import compose_known_exterior
from floorflow.geometry.interior_wall import interior_region
from floorflow.inference.core import semantic_endpoint_labels
from floorflow.evaluation.metrics import floorplan_metrics
from floorflow.model.backbone import FloorplanFlowModel
from floorflow.inference.room_repair import apply_floorplan_repairs
from floorflow.heads.room_groups import room_class_group_index
from floorflow.heads.room_program import room_count_target, room_presence_target
from floorflow.representation.coarse_layout import coarse_room_layout_logits
from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.training.runtime_acceleration import RuntimeConfig, configure_torch_runtime
from floorflow.inference.sampling_policy import SamplingPolicy
from floorflow.inference.structured_decode import decode_structured_floorplan
from floorflow.inference.geometry_wall import apply_wall_face_rooms, geometry_wall_mask
from floorflow.heads.structured_objects import object_group_metrics
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.visualize import colorize, save_comparison_panel
from floorflow.evaluation.full_cli import parse_full_evaluation_args


def parse_args() -> argparse.Namespace:
    return parse_full_evaluation_args()


def _mean_std(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    summary["case_count"] = float(len(rows))
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    runtime = RuntimeConfig(
        precision=precision,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        channels_last=args.channels_last,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    configure_torch_runtime(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    state = (
        checkpoint["ema_model"]
        if args.weight_source == "ema" and checkpoint.get("ema_model")
        else checkpoint["model"]
    )
    taxonomy = (
        FloorplanTaxonomy.from_json(checkpoint["taxonomy"])
        if checkpoint.get("taxonomy")
        else FloorplanTaxonomy.read(args.cache_root / "taxonomy.json")
    )
    image_size = int(args.image_size or config.get("image_size", 256))
    model, config, taxonomy, state = load_checkpoint_model(
        args.checkpoint,
        device,
        taxonomy=taxonomy,
        weight_source=args.weight_source,
    )
    model = runtime.prepare_model(model)
    objective_mode = args.objective_mode or config.get("objective_mode", "velocity")
    wall_model = None
    if args.wall_source_checkpoint is not None:
        wall_model, _, wall_taxonomy, _ = load_checkpoint_model(
            args.wall_source_checkpoint,
            device,
            taxonomy=taxonomy,
        )
        if wall_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("wall-source taxonomy does not match the evaluated checkpoint")
    memory = None
    retrieval_class_weights = None
    if args.retrieval_prior_weight > 0:
        if args.split == "train":
            raise ValueError("retrieval prior must not be evaluated with self-retrieval on train")
        memory = ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=args.retrieval_top_k,
            metric=args.retrieval_metric,
        )
        if args.retrieval_class_weights is not None:
            calibration = json.loads(
                args.retrieval_class_weights.read_text(encoding="utf-8")
            )
            retrieval_class_weights = torch.tensor(
                calibration["weights"],
                device=device,
                dtype=torch.float32,
            )
    opening_threshold = args.opening_threshold
    furniture_threshold = args.furniture_threshold
    if args.threshold_calibration is not None:
        calibration = json.loads(args.threshold_calibration.read_text(encoding="utf-8"))
        opening_threshold = torch.tensor(
            calibration["groups"]["opening"]["thresholds"],
            device=device,
            dtype=torch.float32,
        )
        furniture_threshold = torch.tensor(
            calibration["groups"]["furniture"]["thresholds"],
            device=device,
            dtype=torch.float32,
        )

    full_dataset = CubiCasaFloorplanDataset(
        args.data_root,
        args.split,
        image_size=image_size,
        cache_root=args.cache_root,
        augment=False,
    )
    stop = (
        len(full_dataset)
        if args.limit is None
        else min(len(full_dataset), args.offset + args.limit)
    )
    dataset = Subset(full_dataset, range(args.offset, stop))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **runtime.dataloader_options(device, args.workers),
    )
    rows: list[dict[str, float]] = []
    room_class_intersection = torch.zeros(taxonomy.num_classes, dtype=torch.float64, device=device)
    room_class_union = torch.zeros(taxonomy.num_classes, dtype=torch.float64, device=device)
    partition_intersection = torch.zeros(taxonomy.num_classes, dtype=torch.float64, device=device)
    partition_union = torch.zeros(taxonomy.num_classes, dtype=torch.float64, device=device)
    group_index = room_class_group_index(taxonomy, device)
    group_intersection = torch.zeros(len(group_index.unique()), dtype=torch.float64, device=device)
    group_union = torch.zeros_like(group_intersection)
    examples = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, (target, footprint, floor_ids, names) in enumerate(loader):
        target = target.to(device, non_blocking=True)
        footprint = footprint.to(device, non_blocking=True)
        room_support = None
        raw_objects = None
        room_prior_logits = None
        retrieval_labels = None
        room_program_counts = (
            room_count_target(target, taxonomy)
            if args.room_program_input == "gt_counts"
            else None
        )
        room_program = (
            room_presence_target(target, taxonomy)
            if args.room_program_input in {"gt", "gt_counts"}
            else None
        )
        layout_control = (
            coarse_room_layout_logits(
                target,
                taxonomy,
                size=args.coarse_layout_size,
            )
            if args.coarse_layout_input == "gt"
            else None
        )
        if memory is not None:
            room_prior_logits = memory.retrieve_logits(
                footprint.detach().cpu().numpy(),
                taxonomy,
                device,
                class_weights=retrieval_class_weights,
            )
            if max(
                args.retrieval_opening_weight,
                args.retrieval_furniture_weight,
                args.retrieval_wall_weight,
            ) > 0:
                retrieval_labels = memory.retrieve_labels(
                    footprint.detach().cpu().numpy(),
                    top_k=args.retrieval_top_k,
                ).to(device)
        with runtime.autocast(device):
            if args.structured_output:
                wall_mode = args.wall_mode or (
                    "partition" if args.partition_walls_only else "heads"
                )
                structured_state = None
                if args.structured_flow_steps > 0:
                    structured_state, _trajectory = sample_floorplans(
                        model,
                        footprint,
                        steps=args.structured_flow_steps,
                        heun=not args.no_heun,
                        noise_scale=args.structured_flow_noise_scale,
                        floor_ids=floor_ids.to(device),
                        guidance_scale=args.guidance_scale,
                        objective_mode=objective_mode,
                        semantic_decode_weight=0.0,
                    )
                prediction, room_support, raw_objects, _decoded_objects = decode_structured_floorplan(
                    model,
                    target,
                    footprint,
                    taxonomy,
                    floor_ids=floor_ids.to(device),
                    opening_threshold=opening_threshold,
                    furniture_threshold=furniture_threshold,
                    wall_threshold=args.wall_threshold,
                    object_score_scale=args.object_score_scale,
                    structural_gate=args.structural_gate,
                    use_predicted_walls=not args.partition_walls_only,
                    require_compatible_furniture=not args.allow_incompatible_furniture,
                    partition_wall_radius=args.partition_wall_radius,
                    object_source=args.object_source,
                    opening_source=args.opening_source,
                    furniture_source=args.furniture_source,
                    room_source=args.room_source,
                    return_room_support=True,
                    return_debug=True,
                    room_prior_logits=room_prior_logits,
                    room_prior_weight=args.retrieval_prior_weight,
                    retrieval_labels=retrieval_labels,
                    opening_prior_weight=args.retrieval_opening_weight,
                    furniture_prior_weight=args.retrieval_furniture_weight,
                    wall_prior_weight=args.retrieval_wall_weight,
                    state=structured_state,
                    wall_mode=wall_mode,
                    use_geometry_walls=args.use_geometry_walls,
                    geometry_wall_threshold=args.geometry_wall_threshold,
                    geometry_wall_dilate=args.geometry_wall_dilate,
                    instance_semantic_vote=args.instance_semantic_vote,
                    room_program=room_program,
                    room_program_control=room_program,
                    room_program_counts=room_program_counts,
                    coarse_layout_logits=layout_control,
                    coarse_layout_weight=args.coarse_layout_weight,
                    program_absent_penalty=args.program_absent_penalty,
                )
            else:
                sampled, _trajectory = sample_floorplans(
                    model,
                    footprint,
                    steps=args.steps,
                    heun=not args.no_heun,
                    noise_scale=args.noise_scale,
                    floor_ids=floor_ids.to(device),
                    guidance_scale=args.guidance_scale,
                    objective_mode=objective_mode,
                    semantic_decode_weight=args.semantic_decode_weight,
                )
                prediction = compose_known_exterior(
                    decode_floorplans(sampled, smooth_kernel=args.smooth_kernel),
                    footprint,
                    opening_ids=taxonomy.opening_ids,
                )
        if wall_model is not None:
            geometry_wall = geometry_wall_mask(
                wall_model,
                footprint,
                floor_ids=floor_ids.to(device),
                threshold=args.wall_source_threshold,
                dilate=args.wall_source_dilate,
            )
            if geometry_wall is None:
                wall_labels = semantic_endpoint_labels(
                    wall_model,
                    target,
                    footprint,
                    floor_ids.to(device),
                )
                external_wall = (
                    wall_labels == taxonomy.class_id("wall")
                ) & footprint.bool()
            else:
                external_wall = geometry_wall
            external_wall = external_wall & ~torch.isin(
                prediction,
                torch.tensor(taxonomy.opening_ids, device=device),
            )
            prediction = torch.where(
                external_wall,
                torch.full_like(prediction, taxonomy.class_id("wall")),
                prediction,
            )
            if args.wall_face_rooms:
                prediction = apply_wall_face_rooms(
                    prediction,
                    external_wall,
                    footprint,
                    taxonomy,
                )
        if args.repair_rooms:
            prediction = torch.stack(
                [
                    torch.from_numpy(
                        apply_floorplan_repairs(
                            item.cpu().numpy(),
                            footprint[index].cpu().numpy().astype("uint8"),
                            taxonomy,
                        )
                    ).to(device)
                    for index, item in enumerate(prediction)
                ]
            )
        group_lookup = torch.full(
            (taxonomy.num_classes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        group_lookup[torch.tensor(taxonomy.room_ids, device=device)] = group_index
        predicted_groups = group_lookup[prediction]
        target_groups = group_lookup[target]
        for class_id in taxonomy.room_ids:
            predicted_mask = prediction == class_id
            target_mask = target == class_id
            room_class_intersection[class_id] += (predicted_mask & target_mask).sum()
            room_class_union[class_id] += (predicted_mask | target_mask).sum()
        for group in range(len(group_intersection)):
            predicted_mask = predicted_groups == group
            target_mask = target_groups == group
            group_intersection[group] += (predicted_mask & target_mask).sum()
            group_union[group] += (predicted_mask | target_mask).sum()
        if room_support is not None:
            for class_id in taxonomy.room_ids:
                predicted_mask = room_support == class_id
                target_mask = target == class_id
                partition_intersection[class_id] += (predicted_mask & target_mask).sum()
                partition_union[class_id] += (predicted_mask | target_mask).sum()
        for index, name in enumerate(names):
            metrics = floorplan_metrics(
                prediction[index : index + 1],
                target[index : index + 1],
                footprint[index : index + 1],
                taxonomy,
                room_support=room_support[index : index + 1],
            )
            if room_support is not None:
                metrics["partition_room_macro_iou"] = room_macro_iou_from_labels(
                    room_support[index : index + 1],
                    target[index : index + 1],
                    taxonomy,
                )
            metric_suite = room_metric_suite(
                prediction[index].detach().cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=(
                    (
                        room_support[index]
                        if room_support is not None
                        else torch.isin(
                            prediction[index],
                            torch.tensor(taxonomy.room_ids, device=device),
                        )
                    ).detach().cpu().numpy() > 0
                ),
                predicted_instance_support=torch.isin(
                    prediction[index],
                    torch.tensor(taxonomy.room_ids, device=device),
                ).cpu().numpy(),
                target_room_support=torch.isin(
                    target[index],
                    torch.tensor(taxonomy.room_ids, device=device),
                ).cpu().numpy(),
            )
            instance_metrics = metric_suite["matched_instance"]
            support_instance = metric_suite["support_instance"]
            metrics.update(
                {
                    "instance_geometric_iou": instance_metrics[
                        "matched_geometric_iou"
                    ],
                    "instance_room_f1_at_0_5": instance_metrics[
                        "instance_f1_at_threshold"
                    ],
                    "instance_type_accuracy": instance_metrics[
                        "matched_type_accuracy"
                    ],
                    "instance_count_abs_error": instance_metrics[
                        "count_abs_error"
                    ],
                    "support_instance_geometric_iou": support_instance[
                        "matched_geometric_iou"
                    ],
                    "support_instance_f1_at_0_5": support_instance[
                        "instance_f1_at_threshold"
                    ],
                }
            )
            if raw_objects is not None:
                channel_metrics = object_group_metrics(
                    {
                        key: value[index : index + 1]
                        for key, value in raw_objects.items()
                    },
                    target[index : index + 1],
                    taxonomy,
                )
                metrics.update(
                    {
                        f"channel_{key}": value
                        for key, value in channel_metrics.items()
                    }
                )
                interior = interior_region(footprint[index : index + 1])
                predicted_wall = raw_objects["wall"][index : index + 1].bool() & interior
                expected_wall = (
                    target[index : index + 1] == taxonomy.class_id("wall")
                ) & interior
                metrics["channel_interior_wall_iou"] = float(
                    (predicted_wall & expected_wall).sum().float()
                    / (predicted_wall | expected_wall).sum().clamp_min(1.0).float()
                )
            rows.append(metrics)
            if args.save_rasters:
                stem = name.replace("/", "__").replace("\\", "__")
                raster_dir = args.output_dir / "rasters"
                raster_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raster_dir / f"{stem}.npz",
                    prediction=prediction[index].cpu().numpy().astype(np.uint8),
                    target=target[index].cpu().numpy().astype(np.uint8),
                    footprint=footprint[index].cpu().numpy().astype(np.uint8),
                    room_support=(room_support[index].cpu().numpy().astype(np.uint8) if room_support is not None else np.zeros_like(target[index].cpu().numpy(), dtype=np.uint8)),
                )
            if examples < args.save_examples:
                stem = name.replace("/", "__").replace("\\", "__")
                save_comparison_panel(
                    args.output_dir / "examples" / f"{stem}.png",
                    footprint[index].cpu().numpy().astype(np.uint8),
                    target[index].cpu().numpy(),
                    prediction[index].cpu().numpy(),
                    title=name,
                    taxonomy=taxonomy,
                )
                prediction_image_dir = args.output_dir / "predictions"
                prediction_image_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(colorize(prediction[index].cpu().numpy(), taxonomy)).save(
                    prediction_image_dir / f"{stem}.png"
                )
                examples += 1
        completed = min(len(dataset), (batch_index + 1) * args.batch_size)
        if completed % max(args.batch_size * 10, 1) == 0 or completed == len(dataset):
            print(f"evaluated {completed}/{len(dataset)}", flush=True)

    summary = _mean_std(rows)
    room_class_iou = room_class_intersection / room_class_union.clamp_min(1.0)
    present_room_classes = room_class_union > 0
    summary["room_dataset_macro_iou"] = float(room_class_iou[present_room_classes].mean().cpu())
    summary["room_dataset_micro_iou"] = float(
        room_class_intersection[present_room_classes].sum().cpu()
        / room_class_union[present_room_classes].sum().clamp_min(1.0).cpu()
    )
    group_iou = group_intersection / group_union.clamp_min(1.0)
    summary["room_group_dataset_macro_iou"] = float(group_iou[group_union > 0].mean().cpu())
    if partition_union.any():
        partition_iou = partition_intersection / partition_union.clamp_min(1.0)
        present_partition = partition_union > 0
        summary["partition_room_dataset_macro_iou"] = float(
            partition_iou[present_partition].mean().cpu()
        )
    summary["checkpoint"] = str(args.checkpoint)
    summary["split"] = args.split
    summary["steps"] = args.steps
    summary["heun"] = not args.no_heun
    summary["noise_scale"] = args.noise_scale
    summary["smooth_kernel"] = args.smooth_kernel
    summary["guidance_scale"] = args.guidance_scale
    summary["objective_mode"] = objective_mode
    summary["weight_source"] = args.weight_source
    summary["semantic_decode_weight"] = args.semantic_decode_weight
    summary["repair_rooms"] = args.repair_rooms
    summary["structured_output"] = args.structured_output
    summary["structured_flow_steps"] = args.structured_flow_steps
    summary["structured_flow_noise_scale"] = args.structured_flow_noise_scale
    summary["wall_mode"] = (
        args.wall_mode
        or ("partition" if args.partition_walls_only else "heads")
    )
    summary["object_source"] = args.object_source
    summary["room_source"] = args.room_source
    summary["threshold_calibration"] = (
        str(args.threshold_calibration)
        if args.threshold_calibration is not None
        else None
    )
    summary["wall_source_checkpoint"] = (
        str(args.wall_source_checkpoint)
        if args.wall_source_checkpoint is not None
        else None
    )
    summary["retrieval_prior_weight"] = args.retrieval_prior_weight
    summary["retrieval_top_k"] = args.retrieval_top_k
    summary["retrieval_metric"] = args.retrieval_metric
    summary["retrieval_opening_weight"] = args.retrieval_opening_weight
    summary["retrieval_furniture_weight"] = args.retrieval_furniture_weight
    summary["retrieval_wall_weight"] = args.retrieval_wall_weight
    summary["retrieval_class_weights"] = (
        str(args.retrieval_class_weights)
        if args.retrieval_class_weights is not None
        else None
    )
    summary["retrieval_memory_cases"] = 0 if memory is None else len(memory)
    summary["retrieval_memory_fingerprint"] = None if memory is None else memory.fingerprint
    summary["best_of_n_used"] = False
    summary["room_metric_contract"] = {
        "visible_semantic": "room_macro_iou",
        "instance_geometric": "instance_geometric_iou",
        "occlusion_aware_support": {
            "available": False,
            "reason": "requires polygon-derived room support and occlusion masks",
        },
    }
    summary["sampling_policy"] = SamplingPolicy(
        purpose="single_sample",
        candidates_per_condition=1,
        seed=args.seed,
    ).manifest()
    summary["runtime"] = runtime.manifest()
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "per_case.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
