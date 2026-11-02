"""Metric-aligned room semantics separated from occluded room support."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy


def _case_macro(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    per_case = values.masked_fill(~present, 0.0).sum(dim=1) / (
        present.sum(dim=1).clamp_min(1)
    )
    return per_case.mean()


def visible_room_semantics(
    logits: torch.Tensor,
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CE and Dice over visible room pixels only."""

    room_ids = torch.tensor(
        taxonomy.room_ids,
        device=room_labels.device,
        dtype=torch.long,
    )
    visible = torch.isin(room_labels, room_ids) & (footprint > 0.5)
    lookup = torch.full(
        (taxonomy.num_classes,),
        -1,
        device=room_labels.device,
        dtype=torch.long,
    )
    lookup[room_ids] = torch.arange(len(room_ids), device=room_labels.device)
    target = lookup[room_labels]
    if not visible.any():
        zero = logits.sum() * 0.0
        return zero, {
            "visible_room_ce": zero.detach(),
            "visible_room_dice": zero.detach(),
            "visible_room_pixels": visible.sum().float().detach(),
        }
    ce_pixels = F.cross_entropy(
        logits,
        target.clamp_min(0),
        reduction="none",
    )
    ce_by_case = (
        (ce_pixels * visible).flatten(1).sum(dim=1)
        / visible.flatten(1).sum(dim=1).clamp_min(1)
    )
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(
        target.clamp_min(0),
        logits.shape[1],
    ).permute(0, 3, 1, 2).float()
    mask = visible[:, None].float()
    one_hot = one_hot * mask
    intersection = (probability * one_hot).sum(dim=(2, 3))
    denominator = (probability * mask).sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    present = one_hot.sum(dim=(2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice_by_case = _case_macro(1.0 - dice, present)
    return ce_by_case.mean() + dice_by_case, {
        "visible_room_ce": ce_by_case.mean().detach(),
        "visible_room_dice": dice_by_case.detach(),
        "visible_room_pixels": visible.sum().float().detach(),
    }


def room_support_loss(
    logits: torch.Tensor,
    room_support: torch.Tensor,
    footprint: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Learn room closure under overlays with no semantic identity claim."""

    probability = logits.softmax(dim=1)
    room_probability = probability.sum(dim=1).clamp(1.0e-5, 1.0 - 1.0e-5)
    expected = room_support.to(probability.dtype).clamp(0.0, 1.0)
    inside = (
        footprint[:, 0] if footprint.ndim == 4 else footprint
    ).float()
    inside = (inside > 0.5).float()
    probability = room_probability.float().clamp(1.0e-5, 1.0 - 1.0e-5)
    expected_f = expected.float()
    bce = -(
        expected_f * probability.log()
        + (1.0 - expected_f) * (1.0 - probability).log()
    )
    bce_by_case = (bce * inside).flatten(1).sum(dim=1) / (
        inside.flatten(1).sum(dim=1).clamp_min(1)
    )
    intersection = (room_probability * expected * inside).flatten(1).sum(dim=1)
    denominator = (
        (room_probability + expected) * inside
    ).flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    loss = (bce_by_case + dice).mean()
    return loss, {
        "room_support_bce": bce_by_case.mean().detach(),
        "room_support_dice": dice.mean().detach(),
        "room_support_pixels": (expected * inside).sum().detach(),
    }


def observable_room_loss(
    logits: torch.Tensor,
    room_labels: torch.Tensor,
    room_support: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    support_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    semantic, semantic_components = visible_room_semantics(
        logits,
        room_labels,
        footprint,
        taxonomy,
    )
    support, support_components = room_support_loss(
        logits,
        room_support,
        footprint,
    )
    total = semantic + float(support_weight) * support
    return total, {
        **semantic_components,
        **support_components,
        "observable_room_loss": total.detach(),
    }
