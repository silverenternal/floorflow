"""Functional room groups for coarse-to-fine layout evaluation and losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from floorflow.data.taxonomy import FloorplanTaxonomy

ROOM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("LivingRoom", "Dining", "Den"),
    ("Bedroom", "Room"),
    ("Bath", "Sauna", "Utility"),
    ("Kitchen", "TechnicalRoom"),
    ("Hall", "Entry", "DraughtLobby", "Elevated"),
    ("Closet", "DressingRoom", "Storage"),
    ("Garage", "CarPort"),
    ("Alcove", "Office", "Undefined", "UserDefined", "room_rare"),
)


def room_group_names() -> tuple[str, ...]:
    return (
        "living_dining",
        "sleeping",
        "wet_utility",
        "kitchen_service",
        "circulation",
        "storage",
        "vehicle",
        "other",
    )


def room_class_group_index(
    taxonomy: FloorplanTaxonomy,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map local room-class indices to functional groups."""

    name_to_group = {
        name: group_index
        for group_index, names in enumerate(ROOM_GROUPS)
        for name in names
    }
    output = []
    for class_id in taxonomy.room_ids:
        short = taxonomy.class_names[class_id].replace("room:", "")
        output.append(name_to_group.get(short, len(ROOM_GROUPS) - 1))
    return torch.tensor(output, dtype=torch.long, device=device)


def group_room_logits(
    logits: torch.Tensor,
    group_index: torch.Tensor,
) -> torch.Tensor:
    """Merge fine room logits into functional-group logits."""

    outputs = []
    for group in range(len(ROOM_GROUPS)):
        selected = logits[:, group_index.to(logits.device) == group]
        outputs.append(torch.logsumexp(selected, dim=1) if selected.shape[1] else logits[:, 0] - 1.0e4)
    return torch.stack(outputs, dim=1)


def room_group_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    pixel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Coarse CE + macro Dice over functional room groups."""

    valid = target >= 0
    group_index = room_class_group_index(taxonomy, target.device)
    grouped_logits = group_room_logits(logits, group_index)
    group_target = group_index[target.clamp_min(0)]
    weights = valid.float()
    if pixel_weights is not None:
        weights = weights * pixel_weights.to(device=weights.device, dtype=weights.dtype)
    ce = F.cross_entropy(grouped_logits, group_target, reduction="none")
    ce = (ce * weights).sum() / weights.sum().clamp_min(1.0)
    probability = grouped_logits.softmax(dim=1)
    one_hot = F.one_hot(group_target, grouped_logits.shape[1]).permute(0, 3, 1, 2).float()
    weighted_valid = weights[:, None]
    one_hot = one_hot * weighted_valid
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = (
        (probability * weighted_valid).sum(dim=(0, 2, 3))
        + one_hot.sum(dim=(0, 2, 3))
    )
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice_values = 1.0 - (
        (2.0 * intersection + 1.0) / (denominator + 1.0)
    )
    dice = (dice_values * present).sum() / present.sum().clamp_min(1)
    total = ce + dice
    return total, {
        "room_group_loss": total.detach(),
        "room_group_ce": ce.detach(),
        "room_group_dice": dice.detach(),
    }


@torch.no_grad()
def room_group_macro_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> float:
    group_index = room_class_group_index(taxonomy, target.device)
    lookup = torch.full((taxonomy.num_classes,), -1, dtype=torch.long, device=target.device)
    lookup[torch.tensor(taxonomy.room_ids, device=target.device)] = group_index
    predicted_group = lookup[prediction]
    target_group = lookup[target]
    values = []
    for group in range(len(ROOM_GROUPS)):
        expected = target_group == group
        if not expected.any():
            continue
        predicted = predicted_group == group
        values.append(
            float((predicted & expected).sum().float() / (predicted | expected).sum().clamp_min(1).float())
        )
    return sum(values) / max(1, len(values))
