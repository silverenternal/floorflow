"""Hungarian set loss for semantic element queries."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from floorflow.distribution.element_targets import ElementTargets


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def _center_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    height, width = probability.shape[-2:]
    rows = torch.linspace(0.0, 1.0, height, device=probability.device)
    columns = torch.linspace(0.0, 1.0, width, device=probability.device)
    expected = target.float()

    def _center(values: torch.Tensor) -> torch.Tensor:
        mass = values.sum(dim=(1, 2)).clamp_min(1.0e-6)
        y = (values.sum(dim=2) * rows[None]).sum(dim=1) / mass
        x = (values.sum(dim=1) * columns[None]).sum(dim=1) / mass
        return torch.stack([x, y], dim=-1)

    return (_center(probability) - _center(expected)).abs().mean()


def _diversity_loss(probability: torch.Tensor) -> torch.Tensor:
    flat = probability.flatten(2)
    intersection = flat @ flat.transpose(1, 2)
    denominator = (
        flat.sum(dim=2)[:, :, None]
        + flat.sum(dim=2)[:, None]
        - intersection
    )
    pairwise = intersection / denominator.clamp_min(1.0e-6)
    off_diagonal = ~torch.eye(
        probability.shape[1],
        dtype=torch.bool,
        device=probability.device,
    )
    if probability.shape[1] < 2:
        return probability.sum() * 0.0
    return pairwise[:, off_diagonal].mean()


def group_element_loss(
    prediction: Mapping[str, torch.Tensor],
    target: ElementTargets,
    *,
    boundary_weight: float = 1.0,
    center_weight: float = 0.5,
    diversity_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask_logits = prediction["mask_logits"]
    type_logits = prediction["type_logits"]
    mask_probability = mask_logits.sigmoid()
    type_probability = type_logits.softmax(dim=-1)
    matched_flag = torch.zeros_like(type_logits[..., 0], dtype=torch.bool)
    matched_type = []
    matched_mask = []
    for batch_index in range(mask_logits.shape[0]):
        count = int(target.valid[batch_index].sum())
        if count == 0:
            continue
        target_masks = target.masks[batch_index, :count]
        target_types = target.types[batch_index, :count]
        if target_masks.shape[-2:] != mask_probability.shape[-2:]:
            target_masks = F.interpolate(
                target_masks[:, None],
                size=mask_probability.shape[-2:],
                mode="nearest",
            )[:, 0]
        flat_prediction = mask_probability[batch_index].flatten(1)
        flat_target = target_masks.flatten(1)
        intersection = flat_prediction @ flat_target.T
        denominator = (
            flat_prediction.sum(dim=1)[:, None]
            + flat_target.sum(dim=1)[None]
            - intersection
        )
        mask_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        class_cost = -type_probability[
            batch_index,
            :,
            target_types,
        ]
        cost = (mask_cost + class_cost).detach().cpu().numpy()
        rows, columns = linear_sum_assignment(cost)
        rows_t = torch.as_tensor(rows, device=mask_logits.device)
        columns_t = torch.as_tensor(columns, device=mask_logits.device)
        matched_flag[batch_index, rows_t] = True
        matched_type.append(
            F.cross_entropy(
                type_logits[batch_index, rows_t],
                target_types[columns_t],
            )
        )
        target_mask = target_masks[columns_t]
        bce = F.binary_cross_entropy_with_logits(
            mask_logits[batch_index, rows_t],
            target_mask,
            reduction="none",
        )
        boundary = (
            target_mask
            - F.max_pool2d(-target_mask[:, None], 3, 1, 1)[:, 0]
        ).clamp(0.0, 1.0)
        weights = 1.0 + float(boundary_weight) * boundary
        bce = (bce * weights).sum(dim=(1, 2)) / weights.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        matched_mask.append(
            (
                _dice_loss(mask_logits[batch_index, rows_t], target_mask)
                + 0.2 * bce
                + float(center_weight)
                * _center_loss(
                    mask_probability[batch_index, rows_t],
                    target_mask,
                )
            ).mean()
        )
    diversity = _diversity_loss(mask_probability)
    if matched_type:
        type_loss = torch.stack(matched_type).mean()
        mask_loss = torch.stack(matched_mask).mean()
    else:
        type_loss = mask_logits.sum() * 0.0
        mask_loss = mask_logits.sum() * 0.0
    no_object = torch.full_like(
        type_logits[..., 0],
        type_logits.shape[-1] - 1,
        dtype=torch.long,
    )
    no_object_loss = F.cross_entropy(
        type_logits.transpose(1, 2),
        no_object,
        reduction="none",
    )
    no_object_loss = (
        no_object_loss * (~matched_flag)
    ).sum() / (~matched_flag).sum().clamp_min(1.0)
    presence_logits = prediction["presence_logits"]
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logits,
        matched_flag.float(),
    )
    total = (
        type_loss
        + mask_loss
        + 0.2 * no_object_loss
        + presence_loss
        + float(diversity_weight) * diversity
    )
    return total, {
        "element_type_loss": type_loss.detach(),
        "element_mask_loss": mask_loss.detach(),
        "element_no_object_loss": no_object_loss.detach(),
        "element_presence_loss": presence_loss.detach(),
        "element_mask_diversity": diversity.detach(),
        "element_total_loss": total.detach(),
    }


def element_set_loss(
    predictions: Mapping[str, dict[str, torch.Tensor]],
    targets: Mapping[str, ElementTargets],
    *,
    family_weights: Mapping[str, float] | None = None,
    wall_boundary_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = next(iter(predictions.values()))["mask_logits"].sum() * 0.0
    total = zero
    components: dict[str, torch.Tensor] = {}
    for name, prediction in predictions.items():
        if name not in targets:
            continue
        loss, group_components = group_element_loss(
            prediction,
            targets[name],
            boundary_weight=wall_boundary_weight if name == "wall" else 1.0,
        )
        weight = float(
            (family_weights or {}).get(
                name,
                2.0 if name in {"wall", "opening"} else 1.0,
            )
        )
        total = total + weight * loss
        components[f"{name}_element_loss"] = (weight * loss).detach()
        components.update(
            {f"{name}_{key}": value for key, value in group_components.items()}
        )
    components["element_set_loss"] = total.detach()
    return total, components


__all__ = ["element_set_loss", "group_element_loss"]
