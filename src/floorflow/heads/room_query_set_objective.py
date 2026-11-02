"""Atomic objectives for the room query-set cardinality consumer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from floorflow.objectives.gpu_assignment import exact_assignment_batch
from floorflow.objectives.room_matching_types import RoomQueryMatch
from floorflow.objectives.room_query_matching import (
    match_room_mask_queries,
)
from floorflow.objectives.room_support_assignment import (
    room_support_assignment_loss,
)


def room_query_set_loss(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_valid: torch.Tensor,
    target_masks: torch.Tensor | None = None,
    *,
    existence_weight: float = 0.5,
    count_weight: float = 0.5,
    class_count_weight: float = 0.1,
    class_cardinality_consistency_weight: float = 0.1,
    slot_class_anchor_weight: float = 0.0,
    support_assignment_weight: float = 0.0,
    selection_alignment_weight: float = 0.0,
    presence_rank_weight: float = 0.0,
    presence_rank_margin: float = 0.5,
    presence_count_regression_weight: float = 0.0,
    matches: Sequence[RoomQueryMatch] | None = None,
    return_loss_terms: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Supervise matched-query presence and predicted room cardinality."""

    if "query_existence_logits" not in predictions:
        raise ValueError("query set loss requires query_existence_logits")
    assignment_disagreement = predictions["query_existence_logits"].sum() * 0.0
    if target_masks is not None:
        if target_masks.shape[0] != predictions["mask_logits"].shape[0]:
            raise ValueError("target masks and predictions must share a batch")
        predicted = predictions["mask_logits"].sigmoid().flatten(2)
        targets = target_masks.flatten(2)
        intersection = torch.bmm(predicted, targets.transpose(1, 2))
        denominator = (
            predicted.sum(dim=-1)[:, :, None]
            + targets.sum(dim=-1)[:, None, :]
            - intersection
        )
        iou = intersection / denominator.clamp_min(1.0)
        iou = iou.masked_fill(~target_valid[:, None], 0.0)
        mask_only_target, _mask_only_assignment = (
            _one_to_one_query_targets(
                iou,
                target_valid,
            )
        )
        if matches is not None:
            target, assignment = _shared_query_targets(
                matches,
                predictions["mask_logits"].shape[1],
                device=predictions["query_existence_logits"].device,
            )
        elif "class_logits" in predictions:
            shared_matches = match_room_mask_queries(
                predictions["class_logits"],
                predictions["mask_logits"],
                target_classes,
                target_masks,
                target_valid,
            )
            target, assignment = _shared_query_targets(
                shared_matches,
                predictions["mask_logits"].shape[1],
                device=predictions["query_existence_logits"].device,
            )
        else:
            target = mask_only_target
            assignment = target.bool()
        assignment_disagreement = (
            (target != mask_only_target).float().sum()
            / target_valid.sum().clamp_min(1)
        )
    else:
        target = target_valid.float()
        assignment = target > 0.5
    positive = target.sum().clamp_min(1.0)
    negative = (1.0 - target).sum().clamp_min(1.0)
    positive_weight = (negative / positive).detach().clamp(1.0, 10.0)
    existence = F.binary_cross_entropy_with_logits(
        predictions["query_existence_logits"],
        target,
        pos_weight=positive_weight,
    )
    zero = existence * 0.0
    selection_available = zero
    selected_target_recall = zero
    selected_background_rate = zero
    selection_alignment = zero
    selection = predictions.get("query_count_soft_selection")
    if selection is not None:
        selection_available = zero + 1.0
        if selection.shape != target.shape:
            raise ValueError("query count selection must match [B,Q]")
        selected = selection.float()
        selected_target_recall = (
            (selected * target).sum() / target.sum().clamp_min(1.0)
        )
        selected_background_rate = (
            (selected * (1.0 - target)).sum()
            / (1.0 - target).sum().clamp_min(1.0)
        )
        selection_alignment = (
            1.0 - selected_target_recall + selected_background_rate
        )
    presence_rank = zero
    if float(presence_rank_weight) > 0.0:
        positive_logits = predictions["query_existence_logits"][:, :, None]
        negative_logits = predictions["query_existence_logits"][:, None, :]
        pair_valid = target.bool()[:, :, None] & ~target.bool()[:, None, :]
        pairwise = F.softplus(
            float(presence_rank_margin)
            + negative_logits
            - positive_logits
        )
        presence_rank = (
            pairwise * pair_valid.to(pairwise.dtype)
        ).sum() / pair_valid.sum().clamp_min(1)
    presence_count_regression = zero
    if float(presence_count_regression_weight) > 0.0:
        target_count = target_valid.sum(dim=1).to(
            predictions["query_existence_logits"].dtype
        )
        expected_presence_count = predictions[
            "query_existence_logits"
        ].sigmoid().sum(dim=1)
        presence_count_regression = F.smooth_l1_loss(
            expected_presence_count / target.shape[1],
            target_count / target.shape[1],
        )
    slot_class_anchor = zero
    if float(slot_class_anchor_weight) > 0.0:
        residual = predictions.get("slot_class_residual")
        if residual is None:
            raise ValueError(
                "slot class anchoring requires slot_class_residual"
            )
        slot_class_anchor = residual.float().square().mean()
    support_assignment = zero
    support_components: dict[str, torch.Tensor] = {}
    if float(support_assignment_weight) > 0.0:
        if target_masks is None:
            raise ValueError(
                "support assignment requires target masks"
            )
        owner_probability = predictions.get("slot_owner_probability")
        if owner_probability is None:
            raise ValueError(
                "support assignment requires slot_owner_probability"
            )
        support_assignment, support_components = (
            room_support_assignment_loss(
                owner_probability,
                predictions["class_logits"],
                target_classes,
                target_masks,
                target_valid,
            )
        )
    count = zero
    if "room_count_logits" in predictions:
        target_count = target_valid.sum(dim=1).clamp(
            0,
            predictions["room_count_logits"].shape[-1] - 1,
        )
        count = F.cross_entropy(
            predictions["room_count_logits"],
            target_count,
        )
    class_count = zero
    class_cardinality_consistency = zero
    if "room_class_count_logits" in predictions:
        logits = predictions["room_class_count_logits"]
        target_counts = torch.zeros(
            target_classes.shape[0],
            logits.shape[1],
            dtype=torch.long,
            device=logits.device,
        )
        for batch in range(target_classes.shape[0]):
            valid_classes = target_classes[batch][
                target_valid[batch].bool()
            ].long()
            if valid_classes.numel():
                target_counts[batch] = torch.bincount(
                    valid_classes,
                    minlength=logits.shape[1],
                )[: logits.shape[1]]
        target_counts = target_counts.clamp_max(logits.shape[-1] - 1)
        class_count = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_counts.flatten(),
        )
        expected_counts = (
            logits.float().softmax(dim=-1)
            * torch.arange(
                logits.shape[-1],
                device=logits.device,
                dtype=logits.dtype,
            )
        ).sum(dim=-1)
        class_logits = predictions.get(
            "class_logits",
            predictions.get("room_instance_class_logits"),
        )
        if class_logits is not None:
            if class_logits.shape[-1] - 1 != logits.shape[1]:
                raise ValueError("class and class-count widths must match")
            class_probability = class_logits[..., :-1].float().softmax(dim=-1)
            query_counts = (
                predictions["query_existence_logits"].float().sigmoid()[..., None]
                * class_probability
            ).sum(dim=1)
            class_cardinality_consistency = (
                _normalized_cardinality_distance(
                    query_counts,
                    expected_counts,
                )
            )
    total = (
        float(existence_weight) * existence
        + float(count_weight) * count
        + float(class_count_weight) * class_count
        + float(class_cardinality_consistency_weight)
        * class_cardinality_consistency
        + float(slot_class_anchor_weight) * slot_class_anchor
        + float(support_assignment_weight) * support_assignment
        + float(selection_alignment_weight) * selection_alignment
        + float(presence_rank_weight) * presence_rank
        + float(presence_count_regression_weight)
        * presence_count_regression
    )
    components = {
        "room_query_existence_loss": existence.detach(),
        "room_query_existence_duplicate_rate": _duplicate_assignment_rate(
            assignment,
            target_valid,
        ),
        "room_query_existence_assigned_rate": (
            assignment.float().sum()
            / target_valid.sum().clamp_min(1)
        ).detach(),
        "room_query_mask_only_assignment_disagreement": (
            assignment_disagreement.detach()
        ),
        "room_query_count_selected_target_recall": (
            selected_target_recall.detach()
        ),
        "room_query_count_selected_background_rate": (
            selected_background_rate.detach()
        ),
        "room_query_count_selection_available": (
            selection_available.detach()
        ),
        "room_query_selection_alignment": selection_alignment.detach(),
        "room_query_presence_rank": presence_rank.detach(),
        "room_query_presence_count_regression": (
            presence_count_regression.detach()
        ),
        "room_query_count_loss": count.detach(),
        "room_query_class_count_loss": class_count.detach(),
        "room_query_class_cardinality_consistency": (
            class_cardinality_consistency.detach()
        ),
        "room_query_slot_class_anchor": slot_class_anchor.detach(),
        **support_components,
        "room_query_set_total_loss": total.detach(),
        "room_query_positive_fraction": target.mean().detach(),
        "room_query_presence_target_mean": target.mean().detach(),
    }
    if return_loss_terms:
        return total, components, {
            "query_existence": existence,
            "query_count": count,
            "query_class_count": class_count,
            "query_class_cardinality": class_cardinality_consistency,
            "query_slot_class_anchor": slot_class_anchor,
            "query_support_assignment": support_assignment,
            "query_selection_alignment": selection_alignment,
            "query_presence_rank": presence_rank,
            "query_presence_count": presence_count_regression,
        }
    return total, components


def _normalized_cardinality_distance(
    query_counts: torch.Tensor,
    expected_counts: torch.Tensor,
) -> torch.Tensor:
    """Return a bounded relative L1 distance between two class-count vectors."""

    if query_counts.shape != expected_counts.shape:
        raise ValueError("query and expected class counts must share a shape")
    numerator = (query_counts - expected_counts).abs().sum(dim=-1)
    denominator = 0.5 * (
        query_counts.sum(dim=-1)
        + expected_counts.sum(dim=-1)
    ).clamp_min(1.0)
    return (numerator / denominator).mean()


def _one_to_one_query_targets(
    iou: torch.Tensor,
    target_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match each room target to at most one query slot."""

    if iou.ndim != 3:
        raise ValueError("query IoU must be [B,Q,T]")
    if target_valid.shape != iou.shape[:1] + iou.shape[2:]:
        raise ValueError("target validity must be [B,T]")
    costs = []
    for batch in range(iou.shape[0]):
        cost = -iou[batch].detach().float()
        cost[:, ~target_valid[batch].bool()] = 1.0e4
        costs.append(cost)
    assignments = exact_assignment_batch(costs) if costs else []
    target = iou.new_zeros(
        (iou.shape[0], iou.shape[1]),
        dtype=torch.float32,
    )
    for batch, (rows, columns) in enumerate(assignments):
        valid = target_valid[batch].detach().cpu().numpy()[columns].astype(bool)
        selected_rows = torch.from_numpy(np.asarray(rows)[valid]).to(target.device)
        target[batch].index_fill_(0, selected_rows, 1.0)
    return target.to(iou.dtype), target.bool()


def _duplicate_assignment_rate(
    assignment: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Return duplicate or unmatched assignments per positive target."""

    assigned = assignment.float().sum(dim=1)
    expected = target_valid.float().sum(dim=1)
    return (
        (assigned - expected).abs().sum() / expected.sum().clamp_min(1.0)
    ).detach()


def _shared_query_targets(
    matches: Sequence[RoomQueryMatch],
    num_queries: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one shared matching into a query-presence target."""

    if not matches:
        raise ValueError("shared query matches cannot be empty")
    target = torch.zeros(
        (len(matches), int(num_queries)),
        dtype=torch.float32,
        device=device,
    )
    for batch, match in enumerate(matches):
        rows = torch.as_tensor(
            np.asarray(match.predicted_indices, dtype=np.int64),
            device=target.device,
        )
        if rows.numel():
            target[batch].index_fill_(0, rows.long(), 1.0)
    return target, target.bool()


__all__ = [
    "_normalized_cardinality_distance",
    "_one_to_one_query_targets",
    "_shared_query_targets",
    "room_query_set_loss",
]
