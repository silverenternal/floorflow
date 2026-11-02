"""Reusable room-query blend evaluation for raster-instance training."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floorflow.evaluation.room_metric_suite import room_metric_suite
from floorflow.heads.raster_control_bank import (
    RasterInstanceControl,
)
from floorflow.training.runtime_acceleration import RuntimeConfig


def query_set_route_metrics(
    outputs: dict[str, object],
) -> dict[str, float]:
    """Flatten query-set MIta health into validation rows."""

    diagnostics = outputs.get("query_set_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    values = {
        "query_route_entropy": diagnostics.get("route_entropy"),
        "query_route_attention_entropy": diagnostics.get(
            "route_attention_entropy"
        ),
        "query_route_effective_expert_fraction": diagnostics.get(
            "route_effective_expert_fraction"
        ),
        "query_route_budget": diagnostics.get("route_budget_mean"),
        "query_route_gate_mean": diagnostics.get("route_gate_mean"),
    }
    return {
        name: float(value.detach().float().mean().cpu())
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
        and torch.isfinite(value.detach().float().mean())
    }


def blended_probabilities(
    outputs: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    dense, fused = probability_endpoints(outputs)
    return mix_probability_endpoints(dense, fused, weight)


def probability_endpoints(
    outputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the two expensive softmax tensors once per batch."""

    return (
        outputs["dense_logits"].softmax(dim=1),
        outputs["fused_logits"].softmax(dim=1),
    )


def mix_probability_endpoints(
    dense: torch.Tensor,
    fused: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    return (1.0 - float(weight)) * dense + float(weight) * fused


@torch.inference_mode()
def evaluate(
    model: RasterInstanceControl,
    loader: DataLoader,
    taxonomy,
    device: torch.device,
    weights: tuple[float, ...],
    *,
    return_case_rows: bool = False,
    runtime: RuntimeConfig | None = None,
) -> dict[float, dict[str, float]] | dict[float, list[dict[str, float | str]]]:
    model.eval()
    room_ids = torch.tensor(taxonomy.room_ids, device=device)
    room_ids_np = np.asarray(taxonomy.room_ids, dtype=np.int64)
    rows = {weight: [] for weight in weights}
    autocast = runtime.autocast(device) if runtime is not None else nullcontext()
    for target, footprint, image, _floor_ids, names in loader:
        non_blocking = device.type == "cuda"
        target = target.to(device, non_blocking=non_blocking)
        footprint = footprint.to(device, non_blocking=non_blocking)
        image = image.to(device, non_blocking=non_blocking)
        if runtime is not None and runtime.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        with autocast:
            outputs = model(footprint, control_image=image)
        route_metrics = query_set_route_metrics(outputs)
        dense, fused = probability_endpoints(outputs)
        labels_by_weight = []
        for weight in weights:
            probability = mix_probability_endpoints(dense, fused, weight)
            labels_by_weight.append(
                room_ids[probability.argmax(dim=1)] * footprint.long()
            )
        packed = torch.cat(
            [
                torch.stack(labels_by_weight),
                target.long().unsqueeze(0),
                footprint.long().unsqueeze(0),
            ],
            dim=0,
        )
        host = packed.detach().cpu().numpy()
        labels_np = host[: len(weights)]
        target_np = host[len(weights)]
        footprint_np = host[len(weights) + 1]
        target_support_np = np.isin(target_np, room_ids_np)
        for weight_index, weight in enumerate(weights):
            current_labels = labels_np[weight_index]
            support_np = np.isin(current_labels, room_ids_np)
            for index in range(target.shape[0]):
                suite = room_metric_suite(
                    current_labels[index],
                    target_np[index],
                    footprint_np[index],
                    taxonomy,
                    predicted_room_support=support_np[index],
                    predicted_instance_support=support_np[index],
                    target_room_support=target_support_np[index],
                )
                instance = suite["matched_instance"]
                row = {
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
                        "instance_count_abs_error": instance[
                            "count_abs_error"
                        ],
                        "instance_count_exact": float(
                            instance["count_abs_error"] == 0
                        ),
                        "room_support_iou": suite[
                            "occlusion_aware_support"
                        ]["room_support_iou"],
                        **route_metrics,
                    }
                if return_case_rows:
                    row["case"] = str(names[index])
                rows[weight].append(row)
    if return_case_rows:
        return rows
    return {
        weight: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        }
        for weight, values in rows.items()
    }


def selection_score(
    metrics: dict[str, float],
    *,
    count_weight: float = 0.02,
) -> float:
    return (
        metrics["room_macro_iou"]
        + 0.75 * metrics["instance_room_f1"]
        + 0.1 * metrics["instance_type_accuracy"]
        - float(count_weight) * metrics["instance_count_abs_error"]
    )


def identity_balance_score(
    metrics: dict[str, float],
    reference: dict[str, float],
    *,
    count_weight: float = 0.02,
    type_weight: float = 0.02,
) -> float:
    """Prefer candidates that jointly improve Room IoU and instance F1."""

    room_gain = float(metrics["room_macro_iou"]) - float(
        reference["room_macro_iou"]
    )
    f1_gain = float(metrics["instance_room_f1"]) - float(
        reference["instance_room_f1"]
    )
    count_gain = float(reference["instance_count_abs_error"]) - float(
        metrics["instance_count_abs_error"]
    )
    type_gain = float(metrics["instance_type_accuracy"]) - float(
        reference["instance_type_accuracy"]
    )
    effective_count_weight = min(max(float(count_weight), 0.0), 0.005)
    return (
        min(room_gain, f1_gain)
        + effective_count_weight * count_gain
        + float(type_weight) * type_gain
    )


def resolve_query_blend(
    checkpoint: str | Path,
    override: float | None = None,
    *,
    default: float = 0.5,
) -> tuple[float, str]:
    """Resolve and validate the readout weight used during checkpoint selection."""

    if override is not None:
        weight = float(override)
        source = "override"
    else:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        value = payload.get("blend")
        if value is None:
            value = payload.get("config", {}).get("query_weight")
        if value is None:
            weight = float(default)
            source = "default"
        else:
            weight = float(value)
            source = "checkpoint"
    if not 0.0 <= weight <= 1.0:
        raise ValueError("query blend must be in [0, 1]")
    return weight, source


__all__ = [
    "blended_probabilities",
    "evaluate",
    "identity_balance_score",
    "query_set_route_metrics",
    "resolve_query_blend",
    "selection_score",
]
