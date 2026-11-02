"""Conditional flow-matching losses, integration, and decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.classes import NUM_TARGET_CHANNELS, THIN_STRUCTURE_CLASS_IDS
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.flow.cfm_backend import sample_flow_batch
from floorflow.geometry.core import footprint_band
from floorflow.heads.room_mask_decoder import room_mask_loss, room_mask_partition_loss, room_mask_targets
from floorflow.heads.room_partition import batch_room_partition_target, room_partition_loss
from floorflow.heads.structured_objects import structured_object_loss
from floorflow.heads.topology import topology_aux_loss
from floorflow.model.layoutbridge_adapters import reconstruction_frequency_loss
from floorflow.objectives.semantic_losses import boundary_focal_dice, wall_focal_dice


def target_to_continuous(target: torch.Tensor, num_channels: int = NUM_TARGET_CHANNELS) -> torch.Tensor:
    """Convert labels to foreground one-hot channels; background remains all-zero."""

    foreground = (target - 1).clamp_min(0).long()
    one_hot = F.one_hot(foreground, num_classes=num_channels).permute(0, 3, 1, 2).float()
    foreground_mask = (target > 0).float().unsqueeze(1)
    return one_hot * foreground_mask


def sample_prior(footprint: torch.Tensor, channels: int, noise_scale: float = 0.75) -> torch.Tensor:
    return torch.randn(
        footprint.shape[0],
        channels,
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    ) * footprint.unsqueeze(1) * float(noise_scale)


def sample_simplex_prior(footprint: torch.Tensor, channels: int, concentration: float = 0.5) -> torch.Tensor:
    footprint_bchw = footprint[:, None] if footprint.ndim == 3 else footprint
    distribution = torch.distributions.Dirichlet(
        torch.full(
            (footprint_bchw.shape[0], int(channels) + 1),
            float(concentration),
            device=footprint.device,
            dtype=footprint.dtype,
        )
    )
    return distribution.sample()[:, 1:, None, None] * footprint_bchw


def interpolation(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_view = t[:, None, None, None]
    return (1.0 - t_view) * x0 + t_view * x1


def flow_matching_loss(
    model,
    target: torch.Tensor,
    footprint: torch.Tensor,
    *,
    category_loss_weight: float = 0.20,
    time_sampling: str = "uniform",
    exterior_loss_weight: float = 0.10,
    exterior_band_width: int = 7,
    balanced_object_loss: bool = True,
    max_positive_weight: float = 20.0,
    positive_class_weights: torch.Tensor | None = None,
    floor_ids: torch.Tensor | None = None,
    taxonomy: FloorplanTaxonomy | None = None,
    topology_aux_weight: float = 0.0,
    objective_mode: str = "velocity",
    semantic_head_weight: float = 0.0,
    direct_semantic_weight: float = 0.0,
    boundary_loss_weight: float = 0.0,
    wall_loss_weight: float = 0.0,
    room_partition_weight: float = 0.0,
    frequency_low_weight: float = 0.0,
    frequency_high_weight: float = 0.0,
    room_mask_weight: float = 0.0,
    room_mask_partition_weight: float = 0.0,
    structured_object_weight: float = 0.0,
    semantic_anchor_weight: float = 0.0,
    flow_matcher: str = "legacy",
    flow_path_sigma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x1 = target_to_continuous(target, model.target_channels)
    if objective_mode == "simplex":
        x0 = sample_simplex_prior(footprint, model.target_channels)
    else:
        x0 = sample_prior(footprint, model.target_channels)
    flow = sample_flow_batch(
        x0,
        x1,
        backend=flow_matcher,
        sigma=flow_path_sigma,
        time_sampling=time_sampling,
    )
    t = flow.time
    x_t = flow.location
    velocity = flow.velocity
    footprint_bchw = footprint[:, None] if footprint.ndim == 3 else footprint
    band = footprint_band(footprint_bchw, exterior_band_width)
    # The exterior wall is already known. Keep a small rendering loss, but spend
    # model capacity on rooms, partitions, openings, and furniture inside it.
    pixel_weight = (1.0 - band * (1.0 - float(exterior_loss_weight))) * footprint_bchw
    aux_bundle: dict[str, torch.Tensor] = {}
    anchor_logits = None
    if hasattr(model, "semantic_anchor_logits") and (
        direct_semantic_weight > 0 or semantic_anchor_weight > 0
    ):
        anchor_logits = model.semantic_anchor_logits(x1, footprint, floor_ids=floor_ids)
    if (
        taxonomy is not None
        and (
            topology_aux_weight > 0
            or semantic_head_weight > 0
            or room_partition_weight > 0
            or room_mask_weight > 0
            or room_mask_partition_weight > 0
            or structured_object_weight > 0
        )
        and hasattr(model, "forward_with_aux")
    ):
        raw_prediction, aux_bundle = model.forward_with_aux(
            x_t,
            t,
            footprint,
            floor_ids=floor_ids,
        )
        raw_prediction = raw_prediction * footprint_bchw
    else:
        raw_prediction = model(x_t, t, footprint, floor_ids=floor_ids) * footprint_bchw
    predicted_velocity = raw_prediction
    velocity_error = (predicted_velocity - velocity).pow(2)
    velocity_loss = (velocity_error * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0) / max(
        1, velocity.shape[1]
    )

    predicted_x1 = x_t + (1.0 - t[:, None, None, None]) * predicted_velocity
    anchor_gate = None
    if semantic_anchor_weight > 0 and anchor_logits is not None and hasattr(model, "fuse_semantic_anchor"):
        predicted_x1, anchor_gate = model.fuse_semantic_anchor(predicted_x1, anchor_logits, t)
    positive_error = -x1 * F.logsigmoid(predicted_x1)
    negative_error = -(1.0 - x1) * F.logsigmoid(-predicted_x1)
    category_error = positive_error + negative_error
    if positive_class_weights is not None:
        weights = positive_class_weights.to(device=x1.device, dtype=x1.dtype).view(1, -1, 1, 1)
        category_error = positive_error * weights + negative_error
    elif balanced_object_loss:
        positive_mass = (x1 * pixel_weight).sum(dim=(0, 2, 3))
        negative_mass = ((1.0 - x1) * pixel_weight).sum(dim=(0, 2, 3))
        positive_weight = (negative_mass / positive_mass.clamp_min(1.0)).sqrt().clamp(
            1.0,
            float(max_positive_weight),
        )
        category_error = (
            positive_error * positive_weight.view(1, -1, 1, 1)
            + negative_error
        )
    category_loss = (category_error * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0) / max(
        1, category_error.shape[1]
    )
    total = velocity_loss + float(category_loss_weight) * category_loss
    components = {
        "loss": total.detach(),
        "velocity_loss": velocity_loss.detach(),
        "category_loss": category_loss.detach(),
    }
    if semantic_anchor_weight > 0 and anchor_gate is not None and anchor_logits is not None:
        anchor_loss = F.binary_cross_entropy_with_logits(predicted_x1, x1)
        total = total + float(semantic_anchor_weight) * anchor_loss
        components["semantic_anchor_loss"] = anchor_loss.detach()
        components["semantic_anchor_gate"] = anchor_gate.detach().mean()
        components["loss"] = total.detach()
    if frequency_low_weight > 0 or frequency_high_weight > 0:
        frequency_low, frequency_high = reconstruction_frequency_loss(predicted_x1, x1)
        total = (
            total
            + float(frequency_low_weight) * frequency_low
            + float(frequency_high_weight) * frequency_high
        )
        components["frequency_low_loss"] = frequency_low.detach()
        components["frequency_high_loss"] = frequency_high.detach()
        components["loss"] = total.detach()
    topology_predictions = aux_bundle.get("topology", {}) if aux_bundle else {}
    if topology_predictions and taxonomy is not None:
        auxiliary, aux_components = topology_aux_loss(topology_predictions, target, footprint_bchw, taxonomy)
        total = total + float(topology_aux_weight) * auxiliary
        components["loss"] = total.detach()
        components.update(aux_components)
    semantic_logits = aux_bundle.get("semantic_logits") if aux_bundle else None
    if semantic_logits is not None and semantic_head_weight > 0:
        categorical_logits = torch.cat([torch.zeros_like(semantic_logits[:, :1]), semantic_logits], dim=1)
        if positive_class_weights is not None:
            class_weights = torch.cat(
                [
                    torch.ones(1, device=target.device, dtype=positive_class_weights.dtype),
                    positive_class_weights.to(target.device),
                ]
            )
        else:
            class_weights = None
        semantic_loss = F.cross_entropy(
            categorical_logits,
            target,
            weight=class_weights,
            reduction="mean",
        )
        total = total + float(semantic_head_weight) * semantic_loss
        components["semantic_head_loss"] = semantic_loss.detach()
        if boundary_loss_weight > 0:
            boundary_loss = boundary_focal_dice(semantic_logits, x1)
            total = total + float(boundary_loss_weight) * boundary_loss
            components["boundary_loss"] = boundary_loss.detach()
        if wall_loss_weight > 0 and taxonomy is not None:
            wall_id = taxonomy.class_id("wall")
            if 0 < wall_id <= semantic_logits.shape[1]:
                wall_target = (target == wall_id).float()
                wall_loss = wall_focal_dice(semantic_logits[:, wall_id - 1], wall_target)
                total = total + float(wall_loss_weight) * wall_loss
                components["wall_loss"] = wall_loss.detach()
        components["loss"] = total.detach()
    if direct_semantic_weight > 0 and anchor_logits is not None:
        direct_logits = anchor_logits
        if direct_logits is not None:
            direct_categorical = torch.cat([torch.zeros_like(direct_logits[:, :1]), direct_logits], dim=1)
            direct_loss = F.cross_entropy(
                direct_categorical,
                target,
                weight=class_weights if "class_weights" in locals() else None,
                reduction="mean",
            )
            total = total + float(direct_semantic_weight) * direct_loss
            components["direct_semantic_loss"] = direct_loss.detach()
            components["loss"] = total.detach()
    room_partition_logits = aux_bundle.get("room_partition_logits") if aux_bundle else None
    if room_partition_logits is not None and room_partition_weight > 0 and taxonomy is not None:
        partition_target = batch_room_partition_target(
            target,
            footprint,
            taxonomy,
            fill_occluded=False,
        )
        room_parts = room_partition_loss(room_partition_logits, partition_target)
        total = total + float(room_partition_weight) * room_parts["loss"]
        components["room_partition_loss"] = room_parts["loss"].detach()
        components["room_partition_ce"] = room_parts["partition_ce"]
        components["room_partition_dice"] = room_parts["partition_dice"]
        components["loss"] = total.detach()
    room_mask_predictions = aux_bundle.get("room_mask_predictions") if aux_bundle else None
    if room_mask_predictions is not None and room_mask_weight > 0 and taxonomy is not None:
        num_queries = room_mask_predictions["class_logits"].shape[1]
        target_classes, target_masks, target_valid = room_mask_targets(
            target,
            taxonomy,
            max_rooms=num_queries,
        )
        room_weights = None
        if positive_class_weights is not None:
            room_ids = torch.tensor(taxonomy.room_ids, device=positive_class_weights.device)
            room_weights = positive_class_weights[room_ids - 1]
        room_mask_total, room_mask_components = room_mask_loss(
            room_mask_predictions,
            target_classes,
            target_masks,
            target_valid,
            class_weights=room_weights,
        )
        total = total + float(room_mask_weight) * room_mask_total
        components.update(room_mask_components)
        components["loss"] = total.detach()
    if room_mask_predictions is not None and room_mask_partition_weight > 0 and taxonomy is not None:
        partition_total, partition_components = room_mask_partition_loss(
            room_mask_predictions,
            target,
            footprint,
            taxonomy,
        )
        total = total + float(room_mask_partition_weight) * partition_total
        components.update(partition_components)
        components["loss"] = total.detach()
    structured_predictions = aux_bundle.get("structured_objects") if aux_bundle else None
    if structured_predictions is not None and structured_object_weight > 0 and taxonomy is not None:
        structured_total, structured_components = structured_object_loss(
            structured_predictions,
            target,
            footprint,
            taxonomy,
            positive_class_weights=positive_class_weights,
        )
        total = total + float(structured_object_weight) * structured_total
        components.update(structured_components)
        components["loss"] = total.detach()
    components["objective_mode_simplex"] = torch.tensor(
        float(objective_mode == "simplex"),
        device=x1.device,
    )
    return total, components


@torch.no_grad()
def sample_floorplans(
    model,
    footprint: torch.Tensor,
    *,
    steps: int = 16,
    heun: bool = True,
    noise_scale: float = 0.75,
    floor_ids: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    objective_mode: str = "velocity",
    semantic_decode_weight: float = 0.5,
    style_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the learned velocity field from t=0 to t=1."""

    model.eval()
    footprint = footprint.float()
    mask = footprint.unsqueeze(1)
    if objective_mode == "simplex":
        x = sample_simplex_prior(footprint, model.target_channels)
    else:
        x = sample_prior(footprint, model.target_channels, noise_scale=noise_scale)
    if (
        semantic_decode_weight >= 1.0
        and getattr(model, "use_semantic_endpoint_head", False)
    ):
        _, aux = model.forward_with_aux(
            torch.zeros_like(x),
            torch.zeros(footprint.shape[0], device=footprint.device, dtype=footprint.dtype),
            footprint,
            floor_ids=floor_ids,
        )
        logits = aux.get("semantic_logits")
        if logits is not None:
            probabilities = torch.softmax(
                torch.cat([torch.zeros_like(logits[:, :1]), logits], dim=1),
                dim=1,
            )[:, 1:]
            return probabilities * mask, (probabilities * mask).unsqueeze(1)
    def predict_velocity(state: torch.Tensor, time_value: torch.Tensor, condition: torch.Tensor, floor: torch.Tensor | None) -> torch.Tensor:
        kwargs = {"style_ids": style_ids} if style_ids is not None else {}
        raw = model(
            state,
            time_value,
            condition,
            floor_ids=floor,
            **kwargs,
        ) * mask
        return raw
    trajectory = [x.detach().cpu()]
    null_floor_ids = (
        torch.zeros_like(floor_ids)
        if floor_ids is not None
        else torch.zeros(footprint.shape[0], device=footprint.device, dtype=torch.long)
    )
    steps = max(1, int(steps))
    dt = 1.0 / steps
    for step in range(steps):
        t_value = step * dt
        t = torch.full((footprint.shape[0],), t_value, device=footprint.device, dtype=footprint.dtype)
        velocity = predict_velocity(x, t, footprint, floor_ids)
        if guidance_scale > 1.0:
            unconditional = predict_velocity(x, t, torch.zeros_like(footprint), null_floor_ids)
            velocity = velocity + float(guidance_scale - 1.0) * (velocity - unconditional)
        if heun and step < steps - 1:
            x_pred = (x + dt * velocity) * mask
            t_next = torch.full(
                (footprint.shape[0],),
                min(1.0, t_value + dt),
                device=footprint.device,
                dtype=footprint.dtype,
            )
            velocity_next = predict_velocity(x_pred, t_next, footprint, floor_ids)
            if guidance_scale > 1.0:
                unconditional_next = predict_velocity(x_pred, t_next, torch.zeros_like(footprint), null_floor_ids)
                velocity_next = velocity_next + float(guidance_scale - 1.0) * (
                    velocity_next - unconditional_next
                )
            x = (x + 0.5 * dt * (velocity + velocity_next)) * mask
        else:
            x = (x + dt * velocity) * mask
        if objective_mode == "simplex":
            x = x.clamp_min(0.0) * mask
        if step % 4 == 0 or step == steps - 1:
            trajectory.append(x.detach().cpu())
    if semantic_decode_weight > 0 and getattr(model, "use_semantic_endpoint_head", False):
        logits = (
            model.semantic_anchor_logits(x, footprint, floor_ids=floor_ids)
            if hasattr(model, "semantic_anchor_logits")
            else None
        )
        if logits is not None:
            probabilities = torch.softmax(
                torch.cat([torch.zeros_like(logits[:, :1]), logits], dim=1),
                dim=1,
            )[:, 1:]
            if getattr(model, "use_semantic_anchor_fusion", False) and hasattr(model, "fuse_semantic_anchor"):
                x, _gate = model.fuse_semantic_anchor(
                    x,
                    logits,
                    torch.ones(footprint.shape[0], device=footprint.device, dtype=footprint.dtype),
                )
            else:
                x = ((1.0 - float(semantic_decode_weight)) * x + float(semantic_decode_weight) * probabilities)
            x = x * mask
    return x, torch.stack(trajectory, dim=1)


def decode_floorplans(x1: torch.Tensor, smooth_kernel: int = 1) -> torch.Tensor:
    """Decode foreground channels and an implicit background channel."""

    foreground_scores = x1
    smooth_kernel = max(1, int(smooth_kernel))
    if smooth_kernel > 1:
        smoothed_scores = F.avg_pool2d(
            foreground_scores,
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2,
        )
        # Keep door/window/shower/sauna/structure responses sharp while
        # regularising larger room and cabinet regions.
        for class_id in THIN_STRUCTURE_CLASS_IDS:
            channel = class_id - 1
            if 0 <= channel < foreground_scores.shape[1]:
                smoothed_scores[:, channel] = foreground_scores[:, channel]
        foreground_scores = smoothed_scores
    background_score = 1.0 - foreground_scores.sum(dim=1, keepdim=True).clamp(0.0, 2.0)
    scores = torch.cat([background_score, foreground_scores], dim=1)
    labels = scores.argmax(dim=1).long()
    return labels.clamp(0, scores.shape[1] - 1)
