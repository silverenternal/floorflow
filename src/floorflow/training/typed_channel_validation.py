"""Validation loop for typed-channel experiments with optional feature cache."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floorflow.evaluation.atom_layer_proxies import layer_proxy_metrics
from floorflow.evaluation.conditional_ceiling import room_macro_iou_numpy
from floorflow.evaluation.housemind_structure import housemind_style_metrics
from floorflow.evaluation.room_metric_suite import (
    matched_room_instances,
    room_metric_suite,
)
from floorflow.heads.room_mask_decoder import (
    query_existence_from_class_logits,
    room_mask_metrics,
    room_mask_targets,
)
from floorflow.objectives.room_adjacency import (
    room_adjacency_proxy_metrics,
)
from floorflow.objectives.room_query_separation import (
    count_gated_query_separation,
)
from floorflow.representation.core import extract_room_instances
from floorflow.representation.layered_floorplan import LayeredFloorplan
from floorflow.training.typed_channel_eval import (
    compose_stage_labels,
    family_metrics,
)
from floorflow.training.typed_channel_feature_cache import (
    extract_frozen_features,
    read_frozen_batch,
    write_frozen_batch,
)
from floorflow.training.typed_channel_query_metrics import (
    matched_query_class_metrics,
    multiclass_ece,
    poisson_binomial_count_probability,
    program_assignment_metrics,
    soft_query_mask_iou,
)
from floorflow.training.typed_channel_route_metrics import (
    typed_channel_route_metrics,
)

_poisson_binomial_count_probability = poisson_binomial_count_probability


@torch.no_grad()
def validate_typed_channel(
    model,
    loader: DataLoader,
    taxonomy,
    stage_runtime,
    controls_loader,
    args,
    device: torch.device,
    *,
    epoch: int | None = None,
    row_sink: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    model.eval()
    rows = []
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    non_blocking = device.type == "cuda"
    cache_root = getattr(args, "feature_cache_root", None)
    if getattr(args, "train_scope", "full") in {"full", "field"}:
        cache_root = None
    for target, footprint, floor_ids, names, image in loader:
        target = target.to(device, non_blocking=non_blocking)
        footprint = footprint.to(device, non_blocking=non_blocking)
        cached = (
            read_frozen_batch(cache_root, "val", names, device)
            if cache_root is not None
            else None
        )
        if cached is None:
            controls = controls_loader(
                args.predicted_root,
                "val",
                names,
                args,
                device,
            )
            controls, _stage_diagnostics = stage_runtime(
                controls,
                footprint,
                image.to(device, non_blocking=non_blocking),
                floor_ids.to(device, non_blocking=non_blocking),
            )
            features = extract_frozen_features(model, controls, footprint)
            if cache_root is not None:
                write_frozen_batch(
                    cache_root,
                    "val",
                    names,
                    controls,
                    features,
                )
        else:
            controls, features = cached
        output = model(controls, footprint, cached_features=features)
        room_proxy_available = (
            row_sink is not None
            and "room_instance_mask_logits" in output
            and "room_instance_class_logits" in output
        )
        if room_proxy_available:
            target_classes, target_masks, target_valid = room_mask_targets(
                target,
                taxonomy,
                max_rooms=output["room_instance_mask_logits"].shape[1],
            )
            room_mask_logits = F.interpolate(
                output["room_instance_mask_logits"],
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            room_predictions = {
                "class_logits": output["room_instance_class_logits"],
                "mask_logits": room_mask_logits,
                "query_existence_logits": output.get(
                    "query_existence_logits"
                ),
                "query_count_selection_mask": output.get(
                    "query_count_selection_mask"
                ),
                "aux_mask_logits": [
                    F.interpolate(
                        value,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for value in output["room_instance_aux_mask_logits"]
                ],
                "aux_class_logits": output["room_instance_aux_class_logits"],
            }
            count_targets = target_valid.sum(dim=1)
            query_separation_proxy = tuple(
                count_gated_query_separation(
                    room_predictions["class_logits"][index : index + 1],
                    room_predictions["mask_logits"][index : index + 1],
                    (
                        None
                        if room_predictions["query_existence_logits"] is None
                        else room_predictions["query_existence_logits"][
                            index : index + 1
                        ]
                    ),
                    (
                        None
                        if room_predictions["query_count_selection_mask"] is None
                        else room_predictions["query_count_selection_mask"][
                            index : index + 1
                        ]
                    ),
                )
                for index in range(target.shape[0])
            )
            count_probability = (
                output["room_count_logits"].softmax(dim=-1)
                if "room_count_logits" in output
                else poisson_binomial_count_probability(
                    query_existence_from_class_logits(
                        output["room_instance_class_logits"]
                    )
                )
            )
            count_ece = multiclass_ece(
                count_probability.clamp_min(1.0e-8).log(),
                count_targets,
            )
            count_values = torch.arange(
                count_probability.shape[-1],
                device=device,
                dtype=count_probability.dtype,
            )
            expected_count = (count_probability * count_values).sum(dim=-1)
        rendered, visible = compose_stage_labels(
            output,
            tuple(target.shape[-2:]),
            footprint,
            taxonomy,
            furniture_query_threshold=args.furniture_query_threshold,
        )
        for index in range(target.shape[0]):
            room_prediction = rendered.room_labels[index]
            room_support = torch.isin(room_prediction, room_ids)
            target_support = torch.isin(target[index], room_ids)
            fast_room_macro_iou = room_macro_iou_numpy(
                room_prediction.cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            fast_instance_type_accuracy = _fast_instance_type_accuracy(
                room_prediction.cpu().numpy(),
                target[index].cpu().numpy(),
                taxonomy,
            )
            target_room_pixels = target_support.sum().clamp_min(1)
            room_class_pixel_accuracy = float(
                (
                    (room_prediction == target[index])
                    & target_support
                ).sum()
                / target_room_pixels
            )
            class_recalls = []
            for room_id in room_ids.tolist():
                target_class = target[index] == room_id
                if target_class.any():
                    class_recalls.append(
                        float(
                            (
                                (room_prediction == room_id)
                                & target_class
                            ).sum()
                            / target_class.sum()
                        )
                    )
            room_class_macro_recall = (
                float(np.mean(class_recalls)) if class_recalls else 0.0
            )
            union = (room_support | target_support).sum().clamp_min(1)
            room_union_iou = float(
                (room_support & target_support).sum() / union
            )
            suite = room_metric_suite(
                room_prediction.cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=room_support.cpu().numpy(),
                predicted_instance_support=room_support.cpu().numpy(),
                target_room_support=target_support.cpu().numpy(),
            )
            instance = suite["matched_instance"]
            layer_proxies = layer_proxy_metrics(
                visible[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
                predicted_room_support=(
                    torch.isin(rendered.room_labels[index], room_ids)
                    .cpu()
                    .numpy()
                ),
            )
            graph_metrics = housemind_style_metrics(
                visible[index].cpu().numpy(),
                target[index].cpu().numpy(),
                footprint[index].cpu().numpy(),
                taxonomy,
            )
            family_prediction = LayeredFloorplan(
                room_labels=rendered.room_labels[index : index + 1],
                wall=rendered.wall[index : index + 1],
                opening=rendered.opening[index : index + 1],
                furniture=rendered.furniture[index : index + 1],
                footprint=rendered.footprint[index : index + 1],
                taxonomy=taxonomy,
            )
            family = family_metrics(
                family_prediction,
                target[index : index + 1],
                footprint[index : index + 1],
                taxonomy,
                visible_labels=visible[index : index + 1],
            )
            row = {
                "room_macro_iou": float(
                    suite["visible_semantic"]["room_macro_iou"]
                ),
                "room_f1": float(family["room_f1"]),
                "instance_geometric_iou": float(
                    instance["matched_geometric_iou"]
                ),
                "instance_f1": float(
                    instance["instance_f1_at_threshold"]
                ),
                "instance_type_accuracy": float(
                    instance["matched_type_accuracy"]
                ),
                "instance_count_abs_error": float(
                    instance["count_abs_error"]
                ),
                "room_support_iou": float(
                    suite["occlusion_aware_support"]["room_support_iou"]
                ),
                "query_separation_proxy": float(
                    query_separation_proxy[index]
                ),
                **family,
                **{
                    key: value
                    for key, value in layer_proxies.items()
                    if key.startswith("opening_vector_")
                },
                "node_f1": float(graph_metrics["node_f1"]),
                "node_precision": float(graph_metrics["node_precision"]),
                "node_recall": float(graph_metrics["node_recall"]),
                "edge_overlap": float(graph_metrics["edge_overlap"]),
                "edge_precision": float(graph_metrics["edge_precision"]),
                "edge_recall": float(graph_metrics["edge_recall"]),
                "ged": float(graph_metrics["ged"]),
                **typed_channel_route_metrics(output),
            }
            if "room_adjacency_logits" in output:
                row.update(
                    room_adjacency_proxy_metrics(
                        output["room_adjacency_logits"][index : index + 1],
                        room_mask_logits[index : index + 1],
                    )
                )
            rows.append(row)
            if row_sink is not None and room_proxy_available:
                query_metrics = room_mask_metrics(
                    {
                        "class_logits": room_predictions["class_logits"][
                            index : index + 1
                        ],
                        "mask_logits": room_predictions["mask_logits"][
                            index : index + 1
                        ],
                        "aux_mask_logits": [
                            value[index : index + 1]
                            for value in room_predictions["aux_mask_logits"]
                        ],
                        "aux_class_logits": room_predictions[
                            "aux_class_logits"
                        ][index : index + 1],
                    },
                    target_classes[index : index + 1],
                    target_masks[index : index + 1],
                    target_valid[index : index + 1],
                    taxonomy=taxonomy,
                    target_labels=target[index : index + 1],
                    footprint=footprint[index : index + 1],
                )
                soft_query_iou = soft_query_mask_iou(
                    room_predictions["mask_logits"][index : index + 1],
                    target_masks[index : index + 1],
                    target_valid[index : index + 1],
                )
                query_class_metrics = matched_query_class_metrics(
                    room_predictions["mask_logits"][index : index + 1],
                    room_predictions["class_logits"][index : index + 1],
                    target_classes[index : index + 1],
                    target_masks[index : index + 1],
                    target_valid[index : index + 1],
                )
                program_metrics = program_assignment_metrics(
                    output,
                    index,
                    room_predictions["class_logits"][index : index + 1],
                    room_mask_logits[index : index + 1],
                    target_classes[index : index + 1],
                    target_masks[index : index + 1],
                    target_valid[index : index + 1],
                    controls["program"][index : index + 1],
                )
                dense_instance_logits = F.interpolate(
                    output["room_instance_logits"][index : index + 1],
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                dense_predicted_labels = (
                    room_ids[
                        dense_instance_logits[0].argmax(dim=0)
                    ]
                    * footprint[index].long()
                )
                dense_predicted_count = len(
                    extract_room_instances(
                        dense_predicted_labels.cpu().numpy(),
                        taxonomy,
                        min_area=8,
                    )
                )
                row_sink.append(
                    {
                        "seed": int(args.seed),
                        "checkpoint": f"epoch_{int(epoch or 0):02d}",
                        "case_id": str(names[index]),
                        "proxy_metrics": {
                            "query_mask_dice": float(
                                query_metrics["mask_iou"]
                            ),
                            "matched_class_accuracy": float(
                                query_metrics["matched_class_accuracy"]
                            ),
                            "soft_query_mask_iou": soft_query_iou,
                            "query_separation_proxy": float(
                                query_separation_proxy[index]
                            ),
                            "query_rendered_room_macro_iou": float(
                                query_metrics["rendered_room_macro_iou"]
                            ),
                            "room_union_iou": room_union_iou,
                            "fast_room_macro_iou": fast_room_macro_iou,
                            "room_class_pixel_accuracy": (
                                room_class_pixel_accuracy
                            ),
                            "room_class_macro_recall": (
                                room_class_macro_recall
                            ),
                            "matched_query_class_nll": (
                                query_class_metrics["nll"]
                            ),
                            "matched_query_class_margin": (
                                query_class_metrics["margin"]
                            ),
                            "fast_instance_type_accuracy": (
                                fast_instance_type_accuracy
                            ),
                            "count_distribution_ece": float(
                                count_ece[index]
                            ),
                            "count_expected_abs_error": float(
                                (
                                    expected_count[index]
                                    - count_targets[index]
                                ).abs()
                            ),
                            "count_argmax_abs_error": float(
                                (
                                    count_probability[index].argmax()
                                    - count_targets[index]
                                ).abs()
                            ),
                            "count_target_nll": float(
                                -count_probability[index, count_targets[index]]
                                .clamp_min(1.0e-8)
                                .log()
                            ),
                            "dense_predicted_count_abs_error": abs(
                                dense_predicted_count
                                - int(count_targets[index])
                            ),
                            "program_room_assignment_accuracy": (
                                program_metrics["accuracy"]
                            ),
                            "program_room_assignment_loss": (
                                program_metrics["loss"]
                            ),
                            "wall_pixel_f1": row["wall_f1"],
                            "opening_pixel_f1": row["opening_f1"],
                            "furniture_pixel_f1": row["furniture_f1"],
                            "node_f1": row["node_f1"],
                            "edge_overlap": row["edge_overlap"],
                            "ged": row["ged"],
                            **layer_proxies,
                        },
                        "canonical_metrics": {
                            "room_macro_iou": row["room_macro_iou"],
                            "instance_f1": row["instance_f1"],
                            "instance_count_abs_error": row[
                                "instance_count_abs_error"
                            ],
                            "instance_type_accuracy": row[
                                "instance_type_accuracy"
                            ],
                            "wall_f1": row["wall_f1"],
                            "opening_f1": row["opening_f1"],
                            "furniture_f1": row["furniture_f1"],
                            "node_f1": row["node_f1"],
                            "edge_overlap": row["edge_overlap"],
                            "ged": row["ged"],
                        },
                    }
                )
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _fast_instance_type_accuracy(
    prediction: np.ndarray,
    target: np.ndarray,
    taxonomy,
) -> float:
    predicted = extract_room_instances(prediction, taxonomy, min_area=8)
    expected = extract_room_instances(target, taxonomy, min_area=8)
    matches = matched_room_instances(predicted, expected)
    accepted = [match for match in matches if match.iou >= 0.5]
    return (
        float(np.mean([match.type_correct for match in accepted]))
        if accepted
        else 0.0
    )


__all__ = ["validate_typed_channel"]
