"""Load and read a frozen-feature door/window detection decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.opening_readout import (
    OpeningDetection,
    read_wall_interval_detections,
)
from floorflow.geometry.wall_graph_readout import (
    RasterWallPlan,
    ReadoutOpening,
    raster_labels_to_wall_plan,
)
from floorflow.heads.opening_detection import (
    OpeningDetectionHead,
)
from floorflow.heads.raster_control_bank import (
    load_raster_instance_control_model,
    load_raster_room_model,
)
from floorflow.heads.structured_objects import StructuredObjectHeads


@dataclass
class OpeningHeatmapRuntime:
    """Frozen feature extractor plus a trainable opening detector head."""

    feature_model: torch.nn.Module
    head: torch.nn.Module
    config: dict
    device: torch.device


def encode_opening_features(
    feature_model: torch.nn.Module,
    footprint: torch.Tensor,
    image: torch.Tensor,
    *,
    include_image: bool = False,
    include_room_boundary: bool = False,
) -> torch.Tensor:
    """Extract the shared raster feature used by training and inference."""

    room = getattr(feature_model, "room", feature_model)
    features = room.encode_control(footprint.float(), control_image=image.float())
    if not include_image:
        return features
    footprint_float = (
        footprint.float()
        if footprint.ndim == 4
        else footprint.float()[:, None]
    )
    target_size = tuple(features.shape[-2:])
    if tuple(image.shape[-2:]) != target_size:
        image = F.interpolate(
            image.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    if tuple(footprint_float.shape[-2:]) != target_size:
        footprint_float = F.interpolate(
            footprint_float,
            size=target_size,
            mode="nearest",
        )
    channels = [features, image.float(), footprint_float]
    if include_room_boundary:
        room_output = feature_model(footprint, control_image=image)
        room_logits = (
            room_output["fused_logits"]
            if isinstance(room_output, dict)
            else room_output
        )
        rooms = room_logits.argmax(dim=1)
        boundary = torch.zeros_like(rooms, dtype=torch.bool)
        boundary[:, 1:] |= rooms[:, 1:] != rooms[:, :-1]
        boundary[:, :, 1:] |= rooms[:, :, 1:] != rooms[:, :, :-1]
        channels.append(boundary[:, None].float())
    return torch.cat(channels, dim=1)


def opening_head_outputs(
    head: torch.nn.Module,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the shared detector interface."""

    return head(features)


def opening_detection_maps(
    outputs: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    window_logit_bias: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Separate foreground presence from door/window family scores."""

    if "family_logits" in outputs:
        foreground = outputs["opening_logits"][:, 0].sigmoid()
        family_logits = outputs["family_logits"].clone()
        family_logits[:, 1] += float(window_logit_bias)
        family = family_logits.softmax(dim=1)
        family_probability = foreground[:, None] * family
        family_logits = torch.log(family.clamp_min(1.0e-6))
        maps = {
            "foreground": foreground,
            "family_probability": family_probability,
            "family_logits": family_logits,
        }
        if "opening_center_logits" in outputs:
            maps["center"] = outputs["opening_center_logits"][:, 0].sigmoid()
        if "opening_width_logits" in outputs:
            maps["interval_width"] = outputs["opening_width_logits"][:, 0].sigmoid()
        return maps
    scores = torch.cat(
        [outputs["opening_background_logits"][:, None], outputs["opening_logits"]],
        dim=1,
    ).softmax(dim=1)
    door_indices = [
        index
        for index, class_id in enumerate(taxonomy.opening_ids)
        if taxonomy.class_names[class_id].startswith("door:")
    ]
    window_indices = [
        index + 1
        for index, class_id in enumerate(taxonomy.opening_ids)
        if taxonomy.class_names[class_id].startswith("window:")
    ]
    family = torch.stack(
        [
            scores[:, door_indices].sum(dim=1),
            scores[:, window_indices].sum(dim=1),
        ],
        dim=1,
    )
    foreground = family.max(dim=1).values
    return {
        "foreground": foreground,
        "family_probability": family,
        "family_logits": (
            torch.log(family.clamp_min(1.0e-6))
                + family.new_tensor(
                    [0.0, float(window_logit_bias)]
                )[None, :, None, None]
        ),
    }


def opening_family_probabilities(
    outputs: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Collapse CubiCasa opening subtypes into door/window heatmaps."""

    return opening_detection_maps(outputs, taxonomy)["family_probability"]


def load_opening_heatmap_decoder(
    checkpoint: str | Path,
    taxonomy: FloorplanTaxonomy,
    device: torch.device,
) -> OpeningHeatmapRuntime:
    """Restore a decoder and its frozen raster feature extractor."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_checkpoint = payload.get("feature_checkpoint")
    if feature_checkpoint is None:
        raise ValueError("opening decoder checkpoint has no feature checkpoint")
    feature_payload = torch.load(
        feature_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    feature_state = feature_payload.get("model", {})
    feature_family = str(payload.get("feature_family", ""))
    if not feature_family:
        feature_family = (
            "raster_instance_control"
            if any(key.startswith("room.") for key in feature_state)
            else "raster_room"
        )
    if feature_family == "raster_instance_control":
        feature_model = load_raster_instance_control_model(
            feature_checkpoint,
            taxonomy,
            device,
        )
    elif feature_family == "raster_room":
        feature_model = load_raster_room_model(
            feature_checkpoint,
            taxonomy,
            device,
        )
    else:
        raise ValueError(f"unsupported opening feature family: {feature_family}")
    feature_model.requires_grad_(False).eval()
    in_channels = int(payload.get("head_in_channels", 128))
    head_family = str(payload.get("head_family", "structured_objects"))
    if head_family == "opening_detection_v1":
        hidden_dim = int(payload.get("head_hidden_dim", 96))
        head: torch.nn.Module = OpeningDetectionHead(
            in_channels,
            hidden_dim=hidden_dim,
            variant=str(payload.get("head_variant", "v1")),
            spatial_gap_attention=bool(
                payload.get("spatial_gap_attention", False)
            ),
        ).to(device)
    elif head_family == "structured_objects":
        head = StructuredObjectHeads(in_channels, taxonomy).to(device)
    else:
        raise ValueError(f"unsupported opening head family: {head_family}")
    incompatible = head.load_state_dict(payload["head"], strict=False)
    missing = [
        key for key in incompatible.missing_keys if not key.startswith("gap_")
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "opening checkpoint is incompatible: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    head.eval()
    return OpeningHeatmapRuntime(
        feature_model=feature_model,
        head=head,
        config=dict(payload.get("config", {})),
        device=device,
    )


@torch.inference_mode()
def predict_opening_detection_maps(
    runtime: OpeningHeatmapRuntime,
    footprint: torch.Tensor,
    image: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    window_logit_bias: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return independent foreground and `[B,2,H,W]` family scores."""

    features = encode_opening_features(
        runtime.feature_model,
        footprint,
        image,
        include_image=bool(runtime.config.get("include_image_features", False)),
        include_room_boundary=bool(
            runtime.config.get("include_room_boundary_features", False)
        ),
    )
    outputs = opening_head_outputs(runtime.head, features)
    return opening_detection_maps(
        outputs,
        taxonomy,
        window_logit_bias=window_logit_bias,
    )


@torch.inference_mode()
def predict_opening_family_probabilities(
    runtime: OpeningHeatmapRuntime,
    footprint: torch.Tensor,
    image: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    window_logit_bias: float = 0.0,
) -> torch.Tensor:
    """Return legacy `[B,2,H,W]` foreground-times-family probabilities."""

    return predict_opening_detection_maps(
        runtime,
        footprint,
        image,
        taxonomy,
        window_logit_bias=window_logit_bias,
    )["family_probability"]


def learned_opening_detections_from_maps(
    plan: RasterWallPlan,
    foreground_probability: np.ndarray,
    family_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_area: int = 7,
    family_aggregation: str = "center",
    exterior_wall_window_bias: float = 0.0,
    interval_radius: int = 1,
    score_smoothing: int = 3,
    max_gap_samples: int = 2,
    peak_relative_threshold: float = 0.0,
) -> list[OpeningDetection]:
    """Read intervals from foreground while using family scores only for type."""

    door_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("door:")
    )
    window_id = next(
        class_id
        for class_id in taxonomy.opening_ids
        if taxonomy.class_names[class_id].startswith("window:")
    )
    return read_wall_interval_detections(
        np.asarray(foreground_probability, dtype=np.float32),
        np.asarray(family_logits, dtype=np.float32),
        (door_id, window_id),
        taxonomy.class_names,
        plan,
        threshold=float(threshold),
        radius=int(interval_radius),
        min_width=float(max(3, min_area)),
        max_per_wall=8,
        score_smoothing=int(score_smoothing),
        max_gap_samples=int(max_gap_samples),
        peak_relative_threshold=float(peak_relative_threshold),
        family_aggregation=family_aggregation,
        exterior_wall_window_bias=exterior_wall_window_bias,
    )


def learned_opening_detections(
    plan: RasterWallPlan,
    family_probabilities: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_area: int = 7,
    max_wall_distance: float = 4.0,
) -> list[OpeningDetection]:
    """Detect typed opening spans on an existing wall plan."""

    probability = np.asarray(family_probabilities, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[0] != 2:
        raise ValueError("family probabilities must be [2,H,W]")
    foreground = probability.max(axis=0)
    logits = np.log(np.clip(probability, 1.0e-6, 1.0))
    return learned_opening_detections_from_maps(
        plan,
        foreground,
        logits,
        taxonomy,
        threshold=threshold,
        min_area=min_area,
    )


def attach_learned_opening_detections(
    plan: RasterWallPlan,
    family_probabilities: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_area: int = 7,
    max_wall_distance: float = 4.0,
    host_plan: RasterWallPlan | None = None,
) -> int:
    """Replace only the opening layer while preserving walls and rooms."""

    detections = learned_opening_detections(
        host_plan or plan,
        family_probabilities,
        taxonomy,
        threshold=threshold,
        min_area=min_area,
        max_wall_distance=max_wall_distance,
    )
    plan.openings = [
        ReadoutOpening(
            center=detection.center,
            width=detection.width,
            kind=detection.family,
            wall_index=_nearest_wall_index(detection.center, plan),
        )
        for detection in detections
    ]
    return len(plan.openings)


def attach_learned_opening_detections_from_maps(
    plan: RasterWallPlan,
    foreground_probability: np.ndarray,
    family_logits: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_area: int = 7,
    host_plan: RasterWallPlan | None = None,
    family_aggregation: str = "center",
    exterior_wall_window_bias: float = 0.0,
    interval_radius: int = 1,
    score_smoothing: int = 3,
    max_gap_samples: int = 2,
    peak_relative_threshold: float = 0.0,
    candidate_ranker: Any | None = None,
) -> int:
    """Attach opening intervals decoded from independent score maps."""

    detections = learned_opening_detections_from_maps(
        host_plan or plan,
        foreground_probability,
        family_logits,
        taxonomy,
        threshold=threshold,
        min_area=min_area,
        family_aggregation=family_aggregation,
        exterior_wall_window_bias=exterior_wall_window_bias,
        interval_radius=interval_radius,
        score_smoothing=score_smoothing,
        max_gap_samples=max_gap_samples,
        peak_relative_threshold=peak_relative_threshold,
    )
    if candidate_ranker is not None:
        detections = candidate_ranker.rank(
            detections,
            foreground_probability,
            family_logits,
            host_plan or plan,
            taxonomy,
            frame=float(plan.frame),
        )
    plan.openings = [
        ReadoutOpening(
            center=detection.center,
            width=detection.width,
            kind=detection.family,
            wall_index=_nearest_wall_index(detection.center, plan),
        )
        for detection in detections
    ]
    return len(plan.openings)


def _nearest_wall_index(
    center: tuple[float, float],
    plan: RasterWallPlan,
) -> int:
    if not plan.walls:
        return 0
    point = np.asarray(center, dtype=np.float64)
    best_index = 0
    best_distance = float("inf")
    for index, wall in enumerate(plan.walls):
        start = np.asarray(wall.start, dtype=np.float64)
        end = np.asarray(wall.end, dtype=np.float64)
        delta = end - start
        denominator = float(delta @ delta)
        fraction = (
            0.0
            if denominator <= 1.0e-9
            else float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
        )
        distance = float(np.linalg.norm(point - (start + fraction * delta)))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def plan_with_learned_openings(
    labels: np.ndarray,
    family_probabilities: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    threshold: float = 0.30,
    min_area: int = 7,
    max_wall_distance: float = 4.0,
) -> tuple[RasterWallPlan, int]:
    """Build an explicit wall plan and replace its openings from detection."""

    plan = raster_labels_to_wall_plan(labels, taxonomy)
    count = attach_learned_opening_detections(
        plan,
        family_probabilities,
        taxonomy,
        threshold=threshold,
        min_area=min_area,
        max_wall_distance=max_wall_distance,
    )
    return plan, count


def add_opening_heatmap_arguments(parser: Any) -> None:
    """Register the shared opening-decoder inference options."""

    parser.add_argument("--opening-heatmap-checkpoint", type=Path, default=None)
    parser.add_argument("--opening-heatmap-threshold", type=float, default=0.30)
    parser.add_argument("--opening-heatmap-min-area", type=int, default=7)
    parser.add_argument("--opening-heatmap-max-wall-distance", type=float, default=4.0)
    parser.add_argument(
        "--opening-heatmap-independent-foreground",
        action="store_true",
        help="Use foreground presence independently from door/window family.",
    )
    parser.add_argument(
        "--opening-family-window-bias",
        type=float,
        default=0.0,
        help="Additive calibration bias to the window family logit.",
    )
    parser.add_argument(
        "--opening-family-aggregation",
        choices=("center", "interval"),
        default="center",
    )
    parser.add_argument(
        "--opening-exterior-window-bias",
        type=float,
        default=0.0,
    )
    parser.add_argument("--opening-interval-radius", type=int, default=1)
    parser.add_argument("--opening-score-smoothing", type=int, default=3)
    parser.add_argument("--opening-max-gap-samples", type=int, default=2)
    parser.add_argument(
        "--opening-peak-relative-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--opening-heatmap-candidate-ranker",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--opening-heatmap-candidate-ranker-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--opening-heatmap-min-peak-distance",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--opening-center-decode-mode",
        choices=("wall", "image"),
        default="wall",
    )
    parser.add_argument("--opening-remap-final-wall", action="store_true")


__all__ = [
    "OpeningHeatmapRuntime",
    "add_opening_heatmap_arguments",
    "attach_learned_opening_detections",
    "attach_learned_opening_detections_from_maps",
    "encode_opening_features",
    "learned_opening_detections",
    "learned_opening_detections_from_maps",
    "load_opening_heatmap_decoder",
    "opening_detection_maps",
    "opening_family_probabilities",
    "opening_head_outputs",
    "plan_with_learned_openings",
    "predict_opening_detection_maps",
    "predict_opening_family_probabilities",
]
