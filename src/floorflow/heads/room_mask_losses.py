"""Set-matching and partition losses for the room-mask decoder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.room_mask_area import matched_mask_area_loss
from floorflow.heads.room_mask_fusion import fused_room_mask_logits
from floorflow.heads.room_mask_geometry import (
    matched_mask_centroid_error,
    reference_anchor_loss,
)
from floorflow.heads.room_partition import batch_room_partition_target
from floorflow.objectives.gpu_assignment import assignment_index_tensors
from floorflow.objectives.room_assignment_cost import pairwise_mask_cost
from floorflow.objectives.room_matching_types import RoomQueryMatch
from floorflow.objectives.room_query_matching import (
    match_room_mask_queries,
)
from floorflow.objectives.room_query_separation import (
    active_query_overlap_loss,
    gated_active_query_presence,
    supervised_same_class_overlap_loss,
)


def soft_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> torch.Tensor:
    """Small-object Tversky loss with extra false-positive pressure."""

    probability = logits.sigmoid()
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    index = (true_positive + 1.0) / (true_positive + float(alpha) * false_positive + float(beta) * false_negative + 1.0)
    return (1.0 - index).mean()


def query_partition_loss(
    partition_logits: torch.Tensor,
    binary_mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    assignments: list[tuple[np.ndarray, np.ndarray]],
    *,
    conflict_only: bool = False,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise one mutually exclusive query owner per room pixel."""

    if partition_logits.ndim != 4:
        raise ValueError("partition logits must be [B,Q,H,W]")
    if target_masks.ndim != 4 or target_masks.shape[0] != partition_logits.shape[0]:
        raise ValueError("target masks must be [B,T,H,W]")
    if target_masks.shape[-2:] != partition_logits.shape[-2:]:
        raise ValueError("target and partition spatial sizes must match")
    target = torch.full(
        partition_logits.shape[0:1] + partition_logits.shape[-2:],
        -100,
        dtype=torch.long,
        device=partition_logits.device,
    )
    for batch, (rows, columns) in enumerate(assignments):
        for row, column in zip(rows, columns):
            target[batch, target_masks[batch, int(column)] > 0.5] = int(row)
    valid = target >= 0
    if conflict_only:
        active = (binary_mask_logits.sigmoid() > 0.5).sum(dim=1)
        valid &= active != 1
    cross_entropy = F.cross_entropy(
        partition_logits,
        target.clamp_min(0),
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    valid_weight = valid.to(cross_entropy.dtype)
    cross_entropy = (cross_entropy * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    exact = (
        (partition_logits.argmax(dim=1).eq(target) & valid).to(valid_weight.dtype) * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)
    return cross_entropy, exact.detach(), valid_weight.sum().detach()


def _single_room_mask_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    boundary_weight: float = 2.0,
    assignment_class_weight: float = 1.0,
    assignment_mask_weight: float = 2.0,
    class_loss_weight: float = 1.0,
    mask_loss_weight: float = 2.0,
    program_loss_weight: float = 0.0,
    mask_pos_weight: float = 1.0,
    unmatched_mask_weight: float = 0.0,
    mask_tversky_weight: float = 0.0,
    partition_consistency_weight: float = 0.0,
    partition_conflict_only: bool = False,
    partition_label_smoothing: float = 0.0,
    existence_loss_weight: float = 0.0,
    query_separation_weight: float = 0.0,
    supervised_query_separation: bool = False,
    class_aware_query_separation: bool = True,
    target_counts: Sequence[int] | None = None,
    matches: Sequence[RoomQueryMatch] | None = None,
    reference_anchor_weight: float = 0.0,
    mask_centroid_weight: float = 0.0,
    mask_area_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    class_probability = predictions["class_logits"].softmax(dim=-1)
    mask_probability = predictions["mask_logits"].sigmoid()
    matched_class_losses = []
    matched_mask_losses = []
    unmatched_mask_losses = []
    matched_tversky_losses = []
    matched_dice_values = []
    matched_iou_values = []
    matched_predicted_area_values = []
    matched_target_area_values = []
    matched_centroid_errors = []
    matched_area_losses = []
    reference_anchor_losses = []
    unmatched_predicted_area_values = []
    query_assignments = []
    query_valid_target = torch.zeros_like(class_probability[..., 0])
    weights_all = None
    if class_weights is not None:
        weights_all = torch.cat([class_weights, torch.ones(1, device=class_weights.device, dtype=class_weights.dtype)])
    counts = (
        [int(value) for value in target_counts]
        if target_counts is not None
        else [int(value) for value in target_valid.sum(dim=1).tolist()]
    )

    def _mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return (
            torch.cat([value.flatten() for value in values]).mean()
            if values
            else predictions["mask_logits"].sum() * 0.0
        )

    if matches is not None:
        if len(matches) != target_classes.shape[0]:
            raise ValueError("shared matches must cover the prediction batch")
        pairs = [
            (
                np.asarray(match.predicted_indices, dtype=np.int64),
                np.asarray(match.target_indices, dtype=np.int64),
            )
            for match in matches
        ]
    else:
        shared_matches = match_room_mask_queries(
            predictions["class_logits"],
            predictions["mask_logits"],
            target_classes,
            target_masks,
            target_valid,
            class_cost_weight=float(assignment_class_weight),
            mask_cost_weight=float(assignment_mask_weight),
            target_counts=counts,
        )
        pairs = [
            (
                np.asarray(match.predicted_indices, dtype=np.int64),
                np.asarray(match.target_indices, dtype=np.int64),
            )
            for match in shared_matches
        ]
    query_rows, query_columns, query_match_valid = assignment_index_tensors(
        pairs,
        device=predictions["mask_logits"].device,
    )
    for index in range(target_classes.shape[0]):
        count = counts[index]
        if count == 0:
            if float(unmatched_mask_weight) > 0.0:
                unmatched_logits = predictions["mask_logits"][index]
                unmatched_mask_losses.append(
                    (
                        F.binary_cross_entropy_with_logits(
                            unmatched_logits,
                            torch.zeros_like(unmatched_logits),
                        ),
                        unmatched_logits.new_ones(()),
                    )
                )
            continue
        rows, columns = pairs[index]
        rows_gpu = query_rows[index, query_match_valid[index]]
        columns_gpu = query_columns[index, query_match_valid[index]]
        query_assignments.append((rows, columns))
        matched_class_losses.append(
            F.cross_entropy(
                predictions["class_logits"][index, rows_gpu],
                target_classes[index, columns_gpu],
                weight=weights_all,
            )
        )
        matched_probability = mask_probability[index, rows_gpu]
        matched_logits = predictions["mask_logits"][index, rows_gpu]
        matched_target = target_masks[index, columns_gpu]
        dice = 1.0 - (2.0 * (matched_probability * matched_target).sum(dim=(1, 2)) + 1.0) / (
            matched_probability.sum(dim=(1, 2)) + matched_target.sum(dim=(1, 2)) + 1.0
        )
        matched_dice_values.append(1.0 - dice)
        intersection = (matched_probability * matched_target).sum(dim=(1, 2))
        union = (matched_probability + matched_target - matched_probability * matched_target).sum(dim=(1, 2))
        matched_iou_values.append(intersection / union.clamp_min(1.0))
        matched_predicted_area_values.append(matched_probability.mean(dim=(1, 2)))
        matched_target_area_values.append(matched_target.mean(dim=(1, 2)))
        matched_area_losses.append(matched_mask_area_loss(matched_probability, matched_target))
        matched_centroid_errors.append(matched_mask_centroid_error(matched_probability, matched_target))
        if float(reference_anchor_weight) > 0.0:
            adaptive = predictions.get("adaptive_query_diagnostics") or {}
            reference_points = adaptive.get("reference_samples")
            if not isinstance(reference_points, torch.Tensor):
                raise ValueError("reference anchor supervision requires points")
            reference_anchor_losses.append(reference_anchor_loss(reference_points[index, rows_gpu], matched_target))
        bce = F.binary_cross_entropy_with_logits(
            matched_logits,
            matched_target,
            reduction="none",
            pos_weight=matched_logits.new_tensor(float(mask_pos_weight)),
        )
        eroded = -F.max_pool2d(-matched_target[:, None], 3, stride=1, padding=1)[:, 0]
        boundary = (matched_target - eroded).clamp(0.0, 1.0)
        weights = 1.0 + float(boundary_weight) * boundary
        bce = (bce * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)
        matched_mask_losses.append((dice + 0.2 * bce).mean())
        if float(mask_tversky_weight) > 0.0:
            matched_tversky_losses.append(soft_tversky_loss(matched_logits, matched_target))
        query_valid_target[index].index_fill_(0, rows_gpu, 1.0)
        matched_query = torch.zeros(
            predictions["mask_logits"].shape[1],
            dtype=torch.bool,
            device=predictions["mask_logits"].device,
        )
        matched_query.index_fill_(0, rows_gpu, True)
        unmatched = ~matched_query
        unmatched_logits = predictions["mask_logits"][index]
        unmatched_bce = F.binary_cross_entropy_with_logits(
            unmatched_logits,
            torch.zeros_like(unmatched_logits),
            reduction="none",
        ).mean(dim=(1, 2))
        unmatched_values = unmatched.to(unmatched_bce.dtype)
        if unmatched.any():
            unmatched_predicted_area_values.append(mask_probability[index, unmatched].mean())
        unmatched_mask_losses.append(
            (
                (unmatched_bce * unmatched_values).sum() / unmatched_values.sum().clamp_min(1.0),
                unmatched_values.sum().gt(0).to(unmatched_bce.dtype),
            )
        )
    no_object = torch.full_like(
        predictions["class_logits"][..., 0],
        predictions["class_logits"].shape[-1] - 1,
        dtype=torch.long,
    )
    class_loss = F.cross_entropy(
        predictions["class_logits"].transpose(1, 2),
        no_object,
        weight=weights_all,
        reduction="none",
    )
    class_loss = (class_loss * (1.0 - query_valid_target)).sum() / (1.0 - query_valid_target).sum().clamp_min(1.0)
    if float(existence_loss_weight) > 0.0 and "query_existence_logits" in predictions:
        positive = query_valid_target.sum().clamp_min(1.0)
        negative = (1.0 - query_valid_target).sum().clamp_min(1.0)
        existence_loss = F.binary_cross_entropy_with_logits(
            predictions["query_existence_logits"],
            query_valid_target,
            pos_weight=(negative / positive).clamp(1.0, 10.0),
        )
    else:
        existence_loss = predictions["mask_logits"].sum() * 0.0
    if matched_class_losses:
        class_loss = class_loss + torch.stack(matched_class_losses).mean()
        mask_loss = torch.stack(matched_mask_losses).mean()
    else:
        mask_loss = predictions["mask_logits"].sum() * 0.0
    unmatched_mask_loss = (
        (
            torch.stack([value for value, _present in unmatched_mask_losses]).sum()
            / torch.stack([present for _value, present in unmatched_mask_losses]).sum().clamp_min(1.0)
        )
        if unmatched_mask_losses
        else mask_loss * 0.0
    )
    mask_loss = mask_loss + float(unmatched_mask_weight) * unmatched_mask_loss
    tversky_loss = torch.stack(matched_tversky_losses).mean() if matched_tversky_losses else mask_loss * 0.0
    mask_loss = mask_loss + float(mask_tversky_weight) * tversky_loss
    raw_class_loss = class_loss
    raw_mask_loss = mask_loss
    if program_loss_weight > 0.0:
        foreground_logits = predictions["class_logits"][..., :-1]
        presence_logits = torch.logsumexp(foreground_logits, dim=1)
        target_presence = torch.zeros_like(presence_logits)
        for index, count in enumerate(counts):
            if count:
                target_presence[index, target_classes[index, :count]] = 1.0
        program_loss = F.binary_cross_entropy_with_logits(
            presence_logits,
            target_presence,
            pos_weight=class_weights,
        )
    else:
        program_loss = raw_mask_loss * 0.0
    total = (
        float(class_loss_weight) * raw_class_loss
        + float(mask_loss_weight) * raw_mask_loss
        + float(program_loss_weight) * program_loss
        + float(existence_loss_weight) * existence_loss
    )
    partition_loss = raw_mask_loss * 0.0
    partition_exact = raw_mask_loss.detach() * 0.0
    partition_pixels = raw_mask_loss.detach() * 0.0
    if float(partition_consistency_weight) > 0.0 and "room_partition_logits" in predictions:
        partition_loss, partition_exact, partition_pixels = query_partition_loss(
            predictions["room_partition_logits"],
            predictions["mask_logits"],
            target_masks,
            query_assignments,
            conflict_only=bool(partition_conflict_only),
            label_smoothing=float(partition_label_smoothing),
        )
        total = total + float(partition_consistency_weight) * partition_loss
    separation_loss = raw_mask_loss * 0.0
    reference_anchor_total = _mean_or_zero(reference_anchor_losses)
    total = total + float(reference_anchor_weight) * reference_anchor_total
    mask_centroid_total = _mean_or_zero(matched_centroid_errors)
    total = total + float(mask_centroid_weight) * mask_centroid_total
    mask_area_total = _mean_or_zero(matched_area_losses)
    total = total + float(mask_area_weight) * mask_area_total

    if float(query_separation_weight) > 0.0:
        matches = [
            RoomQueryMatch(
                predicted_indices=np.asarray(rows),
                target_indices=np.asarray(columns),
            )
            for rows, columns in query_assignments
        ]
        separation_loss = (
            supervised_same_class_overlap_loss(
                predictions["mask_logits"],
                matches,
                target_classes,
            )
            if bool(supervised_query_separation)
            else active_query_overlap_loss(
                predictions["mask_logits"],
                gated_active_query_presence(
                    predictions["class_logits"],
                    predictions.get("query_existence_logits"),
                    predictions.get("query_count_selection_mask"),
                ),
                class_probability=(
                    predictions["class_logits"][..., :-1].softmax(dim=-1)
                    if bool(class_aware_query_separation)
                    else None
                ),
            )
        )
        total = total + float(query_separation_weight) * separation_loss
    return total, {
        "room_mask_loss": total.detach(),
        "room_mask_class_loss": raw_class_loss.detach(),
        "room_mask_dice_bce_loss": raw_mask_loss.detach(),
        "room_mask_unmatched_loss": unmatched_mask_loss.detach(),
        "room_mask_tversky_loss": tversky_loss.detach(),
        "_room_mask_class_term": raw_class_loss,
        "_room_mask_mask_term": raw_mask_loss,
        "_room_mask_program_term": program_loss,
        "room_mask_program_loss": program_loss.detach(),
        "room_mask_partition_query_loss": partition_loss.detach(),
        "room_mask_partition_query_exact": partition_exact,
        "room_mask_partition_query_pixels": partition_pixels,
        "room_mask_existence_loss": existence_loss.detach(),
        "room_query_separation_loss": separation_loss.detach(),
        "_room_mask_separation_term": separation_loss,
        "room_mask_matched_dice": _mean_or_zero(matched_dice_values).detach(),
        "room_mask_matched_iou": _mean_or_zero(matched_iou_values).detach(),
        "room_mask_matched_predicted_area": _mean_or_zero(matched_predicted_area_values).detach(),
        "room_mask_matched_target_area": _mean_or_zero(matched_target_area_values).detach(),
        "room_mask_unmatched_predicted_area": _mean_or_zero(unmatched_predicted_area_values).detach(),
        "room_mask_matched_centroid_error": _mean_or_zero(matched_centroid_errors).detach(),
        "room_mask_reference_anchor_loss": reference_anchor_total.detach(),
        "_room_mask_reference_anchor_term": reference_anchor_total,
        "room_mask_centroid_loss": mask_centroid_total.detach(),
        "_room_mask_centroid_term": mask_centroid_total,
        "room_mask_area_loss": mask_area_total.detach(),
        "_room_mask_area_term": mask_area_total,
    }


def room_mask_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    auxiliary_weight: float = 0.5,
    class_weights: torch.Tensor | None = None,
    boundary_weight: float = 2.0,
    assignment_class_weight: float = 1.0,
    assignment_mask_weight: float = 2.0,
    class_loss_weight: float = 1.0,
    mask_loss_weight: float = 2.0,
    program_loss_weight: float = 0.0,
    mask_pos_weight: float = 1.0,
    unmatched_mask_weight: float = 0.0,
    mask_tversky_weight: float = 0.0,
    partition_consistency_weight: float = 0.0,
    partition_conflict_only: bool = False,
    partition_label_smoothing: float = 0.0,
    existence_loss_weight: float = 0.0,
    query_separation_weight: float = 0.0,
    supervised_query_separation: bool = False,
    class_aware_query_separation: bool = True,
    target_counts: Sequence[int] | None = None,
    matches: Sequence[RoomQueryMatch] | None = None,
    reference_anchor_weight: float = 0.0,
    mask_centroid_weight: float = 0.0,
    mask_area_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, components = _single_room_mask_loss(
        predictions,
        target_classes,
        target_masks,
        target_valid,
        class_weights,
        boundary_weight,
        assignment_class_weight,
        assignment_mask_weight,
        class_loss_weight,
        mask_loss_weight,
        program_loss_weight,
        mask_pos_weight,
        unmatched_mask_weight,
        mask_tversky_weight,
        partition_consistency_weight,
        partition_conflict_only,
        partition_label_smoothing,
        existence_loss_weight,
        query_separation_weight,
        supervised_query_separation,
        class_aware_query_separation,
        target_counts,
        matches,
        reference_anchor_weight=reference_anchor_weight,
        mask_centroid_weight=mask_centroid_weight,
        mask_area_weight=mask_area_weight,
    )
    auxiliary_masks = predictions.get("aux_mask_logits", [])
    auxiliary_classes = predictions.get("aux_class_logits", [])
    for level, (mask_logits, class_logits) in enumerate(zip(auxiliary_masks, auxiliary_classes)):
        auxiliary, _aux_components = _single_room_mask_loss(
            {"mask_logits": mask_logits, "class_logits": class_logits},
            target_classes,
            target_masks,
            target_valid,
            class_weights,
            boundary_weight,
            assignment_class_weight,
            assignment_mask_weight,
            class_loss_weight,
            mask_loss_weight,
            program_loss_weight,
            mask_pos_weight,
            unmatched_mask_weight,
            mask_tversky_weight,
            0.0,
            False,
            0.0,
            0.0,
            0.0,
            False,
            True,
            target_counts,
            matches,
            reference_anchor_weight=0.0,
            mask_centroid_weight=0.0,
            mask_area_weight=0.0,
        )
        total = total + float(auxiliary_weight) * auxiliary
        components[f"room_mask_aux_{level}_loss"] = auxiliary.detach()
    components["room_mask_loss"] = total.detach()
    return total, components


def room_mask_partition_loss(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    class_weights: torch.Tensor | None = None,
    pixel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable fusion of query masks into the final room partition."""

    local_target = batch_room_partition_target(target, footprint, taxonomy)
    valid = local_target >= 0
    fusion_mode = "sigmoid_existence" if "query_existence_logits" in predictions else "sigmoid"
    dense_logits = fused_room_mask_logits(predictions, len(taxonomy.room_ids), fusion_mode=fusion_mode)
    ce = F.cross_entropy(
        dense_logits,
        local_target.clamp_min(0),
        weight=class_weights,
        reduction="none",
    )
    weights = valid.float() if pixel_weights is None else pixel_weights.float() * valid.float()
    ce = (ce * weights).sum() / weights.sum().clamp_min(1.0)
    probability = dense_logits.softmax(dim=1)
    one_hot = (
        F.one_hot(
            local_target.clamp_min(0),
            len(taxonomy.room_ids),
        )
        .permute(0, 3, 1, 2)
        .float()
    )
    one_hot = one_hot * weights[:, None]
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (probability * weights[:, None]).sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice_values = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    dice = (dice_values * present).sum() / present.sum().clamp_min(1)
    total = ce + dice
    return total, {
        "room_mask_partition_loss": total.detach(),
        "room_mask_partition_ce": ce.detach(),
        "room_mask_partition_dice": dice.detach(),
    }


__all__ = [
    "pairwise_mask_cost",
    "query_partition_loss",
    "room_mask_loss",
    "room_mask_partition_loss",
    "soft_tversky_loss",
]
