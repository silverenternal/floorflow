#!/usr/bin/env python3
"""Evaluate deterministic element-over-pixel floorplan composition."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from floorflow.data.dataset import (
    CubiCasaFloorplanDataset,
    CubiCasaRasterControlDataset,
)
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.evaluation.distribution_decode_bridge import (
    decode_distribution_candidate,
)
from floorflow.evaluation.distribution_overlay_metrics import (
    constraint_metrics,
    pixel_objects,
    room_metrics,
)
from floorflow.evaluation.fpeval_bridge import fpeval_raster_metrics
from floorflow.evaluation.furniture_query_bridge import (
    apply_furniture_queries,
    load_furniture_query_head,
)
from floorflow.evaluation.overlay_cli import parse_overlay_args
from floorflow.evaluation.overlay_report import build_overlay_report
from floorflow.evaluation.raster_control_runtime import (
    load_raster_room_model,
    raster_control_batch,
)
from floorflow.evaluation.recursive_prior_bridge import (
    predict_base_and_graph_priors,
)
from floorflow.heads.room_groups import room_class_group_index
from floorflow.heads.structured_objects import object_group_metrics
from floorflow.inference.contour_program import (
    load_contour_layout_predictor,
    load_room_program_predictor,
    predict_room_program,
)
from floorflow.inference.distribution_overlay import DistributionOverlaySettings, element_family_scores
from floorflow.inference.dynamic_legalization import legalize_layers
from floorflow.inference.gated_room_decoupling import compose_gated_room_layers
from floorflow.inference.recursive_room_prior import (
    load_graph_adapter,
    load_recursion_policy,
)
from floorflow.inference.room_segmenter_prior import (
    load_room_segmenter_prior,
    predict_room_segmenter_prior,
)
from floorflow.recursion.trust_gate import load_room_graph_trust_gate
from floorflow.representation.contour_memory import ContourPartitionMemory


@torch.inference_mode()
def main() -> None:
    args = parse_overlay_args()
    device = torch.device(args.device)
    model, taxonomy, _ = restore_distribution_model(
        args.checkpoint,
        device,
        dynamic_head_variant=args.dynamic_head_variant,
    )
    model.eval()
    room_group_index = room_class_group_index(taxonomy, device)
    ensemble_model = None
    if args.ensemble_checkpoint is not None:
        ensemble_model, ensemble_taxonomy, _ = restore_distribution_model(
            args.ensemble_checkpoint,
            device,
        )
        if ensemble_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("ensemble taxonomy does not match primary model")
        ensemble_model.eval()
    dataset_class = (
        CubiCasaRasterControlDataset
        if (
            args.room_raster_checkpoint is not None
            and args.room_raster_input_rendering in {"native", "line"}
        )
        else CubiCasaFloorplanDataset
    )
    dataset = dataset_class(
        args.data_root,
        args.split,
        image_size=256,
        cache_root=args.cache_root,
        augment=False,
        **(
            {"image_mode": args.room_raster_input_rendering}
            if dataset_class is CubiCasaRasterControlDataset
            else {}
        ),
    )
    if args.program_checkpoint is None and args.layout_checkpoint is not None:
        raise ValueError("layout prior requires a program checkpoint")
    program_model = None
    layout_model = None
    room_segmenter = None
    room_raster_model = None
    recursion_policy = None
    graph_adapter = None
    trust_gate = None
    furniture_query_head = None
    if args.room_segmenter_checkpoint is not None:
        room_segmenter, _segmenter_config = load_room_segmenter_prior(
            args.room_segmenter_checkpoint,
            taxonomy,
            device,
        )
    if args.room_raster_checkpoint is not None:
        room_raster_model = load_raster_room_model(
            args.room_raster_checkpoint,
            taxonomy,
            device,
        )
    if args.recursion_policy_checkpoint is not None:
        if room_segmenter is None:
            raise ValueError("recursion policy requires a room segmenter")
        recursion_policy = load_recursion_policy(
            args.recursion_policy_checkpoint,
            device,
        )
    if args.recursive_graph_checkpoint is not None:
        if recursion_policy is None:
            raise ValueError("recursive graph requires a recursion policy")
        graph_adapter = load_graph_adapter(
            args.recursive_graph_checkpoint,
            device,
            blend_scale=args.recursive_graph_blend_scale,
        )
    if args.room_graph_trust_checkpoint is not None:
        trust_gate = load_room_graph_trust_gate(
            args.room_graph_trust_checkpoint,
            device,
        )
    if args.furniture_query_checkpoint is not None:
        furniture_query_head, query_taxonomy = load_furniture_query_head(
            args.furniture_query_checkpoint,
            device,
        )
        if query_taxonomy.class_names != taxonomy.class_names:
            raise ValueError("furniture query taxonomy does not match model")
    if args.program_checkpoint is not None:
        program_model = load_room_program_predictor(
            args.program_checkpoint,
            taxonomy,
            device,
        )
        if args.layout_checkpoint is not None:
            layout_model = load_contour_layout_predictor(
                args.layout_checkpoint,
                len(taxonomy.room_ids),
                device,
            )
    stop = min(len(dataset), args.offset + args.limit)
    loader = DataLoader(
        Subset(dataset, range(args.offset, stop)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    settings = DistributionOverlaySettings(
        room_logit_weight=args.overlay_room_logit_weight,
        room_support_threshold=args.overlay_room_support_threshold,
        wall_threshold=args.overlay_wall_threshold,
        opening_threshold=args.overlay_opening_threshold,
        furniture_threshold=args.overlay_furniture_threshold,
        wall_element_weight=args.overlay_wall_element_weight,
        opening_element_weight=args.overlay_opening_element_weight,
        furniture_element_weight=args.overlay_furniture_element_weight,
    )
    use_retrieval = float(args.retrieval_prior_weight) > 0.0
    memory = (
        ContourPartitionMemory.from_cache(
            args.cache_root,
            top_k=args.retrieval_top_k,
            metric=args.retrieval_metric,
        )
        if use_retrieval
        else None
    )
    retrieval_weights = (
        torch.tensor(
            json.loads(
                args.retrieval_weights_report.read_text(encoding="utf-8")
            )["weights"],
            device=device,
        )
        if use_retrieval
        else None
    )
    rows = {"raw": [], "element_room": [], "overlay": []}
    object_rows = {"raw": [], "overlay": []}
    constraint_rows = []
    opening_readout_counts = Counter()
    room_inter = np.zeros(len(taxonomy.room_ids), dtype=np.int64)
    room_union = np.zeros(len(taxonomy.room_ids), dtype=np.int64)
    room_support_inter = 0
    room_support_union = 0
    case_index = args.offset
    overlay_case_rows = []
    for batch in loader:
        if (
            args.room_raster_checkpoint is not None
            and args.room_raster_input_rendering in {"native", "line"}
        ):
            target, footprint, control_image, floor_ids, names = batch
        else:
            target, footprint, floor_ids, names = batch
            control_image = None
        target = target.to(device)
        footprint = footprint[:, None].to(device)
        style_ids = (
            torch.full(
                (footprint.shape[0],),
                int(args.style_id),
                device=device,
                dtype=torch.long,
            )
            if args.style_id is not None
            else None
        )
        room_prior_logits = None
        fallback_room_prior_logits = None
        base_room_prior_logits = None
        graph_room_prior_logits = None
        recursive_decision = None
        decoupling_diagnostics = None
        program = None
        if program_model is not None:
            program = predict_room_program(
                *program_model,
                footprint,
                floor_ids.to(device),
            )
            if layout_model is not None:
                room_prior_logits = layout_model(
                    footprint,
                    program.to(footprint.dtype),
                )
        if room_segmenter is not None:
            direct_prior = predict_room_segmenter_prior(
                room_segmenter,
                footprint,
            )
            room_prior_logits = direct_prior
            base_room_prior_logits = direct_prior
            if recursion_policy is not None:
                (
                    base_room_prior_logits,
                    graph_room_prior_logits,
                    recursive_decision,
                ) = predict_base_and_graph_priors(
                    segmenter=room_segmenter,
                    policy=recursion_policy,
                    graph_adapter=graph_adapter,
                    footprint=footprint,
                    direct_prior=direct_prior,
                    args=args,
                    room_group_index=room_group_index,
                )
                room_prior_logits = base_room_prior_logits
            fallback_room_prior_logits = room_prior_logits
        if room_raster_model is not None:
            control_image = raster_control_batch(
                target,
                footprint,
                control_image,
                taxonomy,
                device,
            )
            raster_logits = room_raster_model(
                footprint,
                control_image=control_image.to(device),
            )
            if room_prior_logits is None:
                room_prior_logits = raster_logits
            else:
                blend = float(args.room_raster_blend)
                room_prior_logits = (
                    (1.0 - blend)
                    * room_prior_logits.log_softmax(dim=1)
                    + blend * raster_logits.log_softmax(dim=1)
                )
        decoded = decode_distribution_candidate(
            model,
            memory,
            retrieval_weights,
            footprint,
            taxonomy,
            args=args,
            floor_ids=floor_ids.to(device),
            style_ids=style_ids,
            room_prior_logits=room_prior_logits,
            program=program,
            ensemble_model=ensemble_model,
            settings=settings,
        )
        layers = decoded["layers"]
        if (
            args.room_raster_support_fallback
            and room_raster_model is not None
            and fallback_room_prior_logits is not None
        ):
            fallback_decoded = decode_distribution_candidate(
                model,
                memory,
                retrieval_weights,
                footprint,
                taxonomy,
                args=args,
                floor_ids=floor_ids.to(device),
                style_ids=style_ids,
                room_prior_logits=fallback_room_prior_logits,
                program=program,
                ensemble_model=ensemble_model,
                settings=settings,
            )
            fallback_rooms = fallback_decoded["layers"].room_labels
            fallback_support = torch.isin(
                fallback_decoded["labels"],
                torch.tensor(
                    taxonomy.room_ids,
                    device=fallback_rooms.device,
                ),
            )
            use_raster = fallback_support & (layers.room_labels != 0)
            layers.room_labels = torch.where(
                use_raster,
                layers.room_labels,
                fallback_rooms,
            )
        if graph_room_prior_logits is not None:
            graph_decoded = decode_distribution_candidate(
                model,
                memory,
                retrieval_weights,
                footprint,
                taxonomy,
                args=args,
                floor_ids=floor_ids.to(device),
                style_ids=style_ids,
                room_prior_logits=graph_room_prior_logits,
                program=program,
                ensemble_model=ensemble_model,
                settings=settings,
            )
            layers, decoupling_diagnostics = compose_gated_room_layers(
                layers,
                graph_decoded["layers"],
                base_room_prior_logits,
                graph_room_prior_logits,
                footprint[:, 0],
                taxonomy,
                graph_confidence_threshold=(
                    args.room_graph_confidence_threshold
                ),
                trust_gate=trust_gate,
                trust_threshold=args.room_graph_trust_threshold,
            )
        if furniture_query_head is not None:
            layers, _furniture_query_diagnostics = apply_furniture_queries(
                furniture_query_head,
                decoded["distribution"],
                layers,
                footprint,
                taxonomy,
                threshold=args.furniture_query_threshold,
            )
        if args.legalize_objects:
            layers = legalize_layers(
                layers,
                footprint[:, 0],
                taxonomy,
                opening_wall_radius=args.legalization_opening_wall_radius,
                closure_radius=args.legalization_closure_radius,
                dynamic_closure=args.legalization_dynamic_closure,
            )
        overlay_labels = layers.visible_labels()
        element_scores = element_family_scores(
            decoded["distribution"]["elements"],
            model.hub.hierarchy,
            overlay_labels.shape[-2:],
        )
        opening_scores = element_scores.get("opening")
        raw_rooms = decoded["raw_rooms"]
        element_rooms = decoded["element_rooms"]
        raw_pixel_labels = decoded["pixel_labels"]
        if args.save_labels_dir is not None:
            args.save_labels_dir.mkdir(parents=True, exist_ok=True)
            for index in range(target.shape[0]):
                np.save(
                    args.save_labels_dir / f"case_{case_index + index:04d}.npy",
                    overlay_labels[index].cpu().numpy().astype(np.uint8),
                )
        case_index += target.shape[0]
        for index in range(target.shape[0]):
            constraint_rows.append(
                constraint_metrics(
                    overlay_labels[index],
                    target[index],
                    footprint[index, 0],
                    layers.wall[index],
                    layers.opening[index],
                    layers.furniture[index],
                    taxonomy,
                    predicted_room_support=torch.isin(
                        layers.room_labels[index],
                        torch.tensor(
                            taxonomy.room_ids,
                            device=layers.room_labels.device,
                        ),
                    ),
                )
            )
        object_rows["raw"].append(
            object_group_metrics(
            pixel_objects(raw_pixel_labels, taxonomy),
                target,
                taxonomy,
            )
        )
        object_rows["overlay"].append(
            object_group_metrics(
                {
                    "wall": layers.wall,
                    "opening": layers.opening,
                    "furniture": layers.furniture,
                },
                target,
                taxonomy,
            )
        )
        for index in range(target.shape[0]):
            rows["raw"].append(
                room_metrics(
                    raw_rooms[index],
                    target[index],
                    footprint[index, 0],
                    raw_rooms[index],
                    taxonomy,
                )
            )
            rows["element_room"].append(
                room_metrics(
                    element_rooms[index],
                    target[index],
                    footprint[index, 0],
                    element_rooms[index],
                    taxonomy,
                )
            )
            rows["overlay"].append(
                room_metrics(
                    overlay_labels[index],
                    target[index],
                    footprint[index, 0],
                    layers.room_labels[index],
                    taxonomy,
                )
            )
            fpeval = fpeval_raster_metrics(
                overlay_labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
                prediction_opening_probability=(
                    opening_scores[index].sum(dim=0).cpu().numpy()
                    if opening_scores is not None
                    else None
                ),
                prediction_opening_logits=(
                    opening_scores[index].cpu().numpy()
                    if opening_scores is not None
                    else None
                ),
                prediction_opening_threshold=args.overlay_opening_threshold,
            )
            opening_readout_counts[
                str(fpeval.get("opening_readout_source", "labels"))
            ] += 1
            summary = fpeval["summary"]["0.05"]
            room_fpeval = fpeval_raster_metrics(
                overlay_labels[index].cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
                tolerance_fractions=(0.05,),
                explicit_prediction_rooms=True,
                explicit_target_rooms=True,
            )
            predicted_rooms = np.isin(
                overlay_labels[index].cpu().numpy(),
                taxonomy.room_ids,
            )
            target_rooms = np.isin(
                target[index].cpu().numpy(),
                taxonomy.room_ids,
            )
            room_support_inter += int(
                np.logical_and(predicted_rooms, target_rooms).sum()
            )
            room_support_union += int(
                np.logical_or(predicted_rooms, target_rooms).sum()
            )
            for room_index, class_id in enumerate(taxonomy.room_ids):
                predicted = overlay_labels[index].cpu().numpy() == class_id
                expected = target[index].cpu().numpy() == class_id
                room_inter[room_index] += int(
                    np.logical_and(predicted, expected).sum()
                )
                room_union[room_index] += int(
                    np.logical_or(predicted, expected).sum()
                )
            rows["overlay"][-1].update(
                {
                    "fpeval_wall_f1": summary["wall_f1"],
                    "wall_graph_f1": summary["wall_f1"],
                    "fpeval_opening_f1": summary["opening_f1"],
                    "fpeval_room_face_f1": summary["room_f1"],
                    "fpeval_room_polygon_f1": room_fpeval["summary"]["0.05"][
                        "room_f1"
                    ],
                    "fpeval_edit_cost": summary["edit_cost"],
                    "fpeval_validity": float(
                        fpeval["prediction_validity"]["valid"]
                    ),
                    "fpeval_watertight": float(
                        fpeval["prediction_validity"]["watertight"]
                    ),
                    "target_watertight": float(
                        fpeval["target_validity"]["watertight"]
                    ),
                }
            )
            overlay_case_rows.append(
                {
                    "name": str(names[index]),
                    "recursive_decision": (
                        {
                            "action": recursive_decision["actions"][index],
                            "strength": recursive_decision["strengths"][index],
                            "depths": recursive_decision["depths"][index],
                            "graph": recursive_decision["graph"][index],
                        }
                        if recursive_decision is not None
                        else None
                    ),
                    "room_gate": decoupling_diagnostics,
                    **rows["overlay"][-1],
                }
            )
    report = build_overlay_report(
        args=args,
        overlay_settings=settings,
        stop=stop,
        rows=rows,
        object_rows=object_rows,
        constraint_rows=constraint_rows,
        opening_readout_counts=opening_readout_counts,
        room_inter=room_inter,
        room_union=room_union,
        room_support_inter=room_support_inter,
        room_support_union=room_support_union,
    )
    if args.per_case_output is not None:
        args.per_case_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_case_output.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "offset": args.offset,
                    "cases": overlay_case_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
