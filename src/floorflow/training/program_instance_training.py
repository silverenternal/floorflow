"""Hungarian and schedule losses for program-to-instance assignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.heads.room_mask_losses import pairwise_mask_cost
from floorflow.objectives.room_matching_types import RoomQueryMatch


def program_instance_assignment_loss(
    assignment_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    program: torch.Tensor,
    *,
    schedule_weight: float = 0.25,
    matches: list[RoomQueryMatch] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match queries to target rooms and supervise class plus program totals."""

    if assignment_logits.shape[1] != mask_logits.shape[1]:
        raise ValueError("assignment and mask query counts differ")
    if assignment_logits.shape[-1] != program.shape[1] + 1:
        raise ValueError("assignment classes do not match program width")
    if mask_logits.shape[-2:] != target_masks.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if matches is not None and len(matches) != assignment_logits.shape[0]:
        raise ValueError("shared room matches must cover the batch")
    mask_probability = mask_logits.sigmoid()
    losses = []
    correct = 0
    matched_count = 0
    for index in range(assignment_logits.shape[0]):
        count = int(target_valid[index].sum())
        if count == 0:
            continue
        if matches is None:
            cost = pairwise_mask_cost(
                mask_probability[index],
                target_masks[index, :count],
            )
            rows, columns = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
        else:
            rows = matches[index].predicted_indices
            columns = matches[index].target_indices
        if len(rows) != count or len(columns) != count:
            raise ValueError("shared room matching must cover every valid target")
        row_tensor = torch.as_tensor(
            rows,
            device=assignment_logits.device,
        )
        column_tensor = torch.as_tensor(
            columns,
            device=assignment_logits.device,
        )
        target = target_classes[index, column_tensor]
        class_logits = assignment_logits[index, row_tensor, 1:]
        losses.append(F.cross_entropy(class_logits, target))
        correct += int((class_logits.argmax(dim=-1) == target).sum())
        matched_count += len(rows)
    assignment_loss = (
        torch.stack(losses).mean()
        if losses
        else assignment_logits.sum() * 0.0
    )
    foreground = assignment_logits.softmax(dim=-1)[..., 1:]
    expected_count = foreground.sum(dim=1)
    # log(1 - exp(-x)) is the logit of the Poisson presence probability. Keep
    # the whole path logit-safe so autocast never calls unsafe BCE on sigmoid.
    presence_logits = torch.log(
        -torch.expm1(-expected_count.clamp_min(1.0e-6))
    )
    schedule_loss = F.binary_cross_entropy_with_logits(
        presence_logits,
        program.float(),
    )
    total = assignment_loss + float(schedule_weight) * schedule_loss
    return total, {
        "program_instance_assignment_loss": assignment_loss.detach(),
        "program_instance_schedule_loss": schedule_loss.detach(),
        "program_instance_assignment_accuracy": (
            assignment_logits.new_tensor(correct / max(1, matched_count))
        ),
    }


__all__ = ["program_instance_assignment_loss"]
