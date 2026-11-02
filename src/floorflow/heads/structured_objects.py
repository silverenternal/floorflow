"""Object-group semantic heads for walls, openings, and furniture."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.geometry.interior_wall import interior_wall_loss
from floorflow.objectives.semantic_losses import dice_loss


def initialize_object_priors(
    heads: StructuredObjectHeads,
    class_counts: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Initialize null logits from actual foreground pixel frequencies."""

    counts = torch.as_tensor(class_counts, dtype=torch.float64).cpu()
    total = counts.sum().clamp_min(1.0)
    priors: dict[str, float] = {}
    for name, class_ids in (
        ("opening", heads.opening_ids),
        ("furniture", heads.furniture_ids),
    ):
        probability = float(counts[class_ids.cpu()].sum() / total)
        probability = min(0.49, max(2.0e-4, probability))
        class_count = max(1, len(class_ids))
        background_bias = float(np.log((1.0 - probability) / probability * class_count))
        background = getattr(heads, f"{name}_background")
        with torch.no_grad():
            background.bias.fill_(background_bias)
            background.weight.zero_()
        priors[f"{name}_prior"] = probability
    wall_probability = float(counts[heads.wall_id] / total)
    wall_probability = min(0.49, max(2.0e-4, wall_probability))
    with torch.no_grad():
        heads.wall.bias.fill_(float(np.log(wall_probability / (1.0 - wall_probability))))
        heads.wall.weight.zero_()
    priors["wall_prior"] = wall_probability
    return priors


class StructuredObjectHeads(nn.Module):
    """Shared spatial trunk with independent object-family classifiers."""

    def __init__(self, in_channels: int, taxonomy: FloorplanTaxonomy) -> None:
        super().__init__()
        hidden = max(48, int(in_channels))
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(max(1, hidden // 16), hidden),
            nn.SiLU(),
        )
        self.opening = nn.Conv2d(hidden, len(taxonomy.opening_ids), 1)
        self.furniture = nn.Conv2d(hidden, len(taxonomy.furniture_ids), 1)
        self.opening_background = nn.Conv2d(hidden, 1, 1)
        self.furniture_background = nn.Conv2d(hidden, 1, 1)
        self.wall = nn.Conv2d(hidden, 1, 1)
        for head in (
            self.opening,
            self.furniture,
            self.opening_background,
            self.furniture_background,
            self.wall,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("opening_ids", torch.tensor(taxonomy.opening_ids), persistent=False)
        self.register_buffer("furniture_ids", torch.tensor(taxonomy.furniture_ids), persistent=False)
        self.wall_id = taxonomy.class_id("wall")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features)
        return {
            "opening_logits": self.opening(hidden),
            "furniture_logits": self.furniture(hidden),
            "opening_background_logits": self.opening_background(hidden).squeeze(1),
            "furniture_background_logits": self.furniture_background(hidden).squeeze(1),
            "wall_logits": self.wall(hidden).squeeze(1),
        }


def _group_target(
    target: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    lookup_size = max(int(class_ids.max().item()), int(target.max().item())) + 1
    lookup = torch.full((lookup_size,), -100, dtype=torch.long, device=target.device)
    lookup[class_ids] = torch.arange(len(class_ids), device=target.device)
    return lookup[target]


def group_ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_ids: torch.Tensor,
    footprint: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    background_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_target = _group_target(target, class_ids)
    target_for_ce = torch.where(local_target >= 0, local_target + 1, 0)
    valid = footprint.bool()
    background = (
        background_logits[:, None]
        if background_logits is not None
        else torch.zeros_like(logits[:, :1])
    )
    scores = torch.cat([background, logits], dim=1)
    weights = None
    if class_weights is not None:
        weights = torch.cat([torch.ones(1, device=class_weights.device), class_weights])
    ce = F.cross_entropy(scores, target_for_ce, weight=weights, reduction="none")
    ce = ce[valid].mean()
    probabilities = logits.softmax(dim=1) * valid[:, None].float()
    one_hot = F.one_hot(local_target.clamp_min(0), logits.shape[1]).permute(0, 3, 1, 2).float()
    present = ((local_target >= 0) & valid).any(dim=(0, 1, 2))
    if present.any():
        one_hot = one_hot * (local_target >= 0).unsqueeze(1).float()
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice_component = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))[present].mean()
    else:
        dice_component = probabilities.sum() * 0.0
    return ce + dice_component, ce.detach(), dice_component.detach()


def wall_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    wall_id: int,
    footprint: torch.Tensor,
) -> torch.Tensor:
    expected = (target == wall_id).float()
    positive = expected.sum().clamp_min(1.0)
    negative = (footprint.bool() & ~expected.bool()).sum().clamp_min(1.0)
    positive_weight = (negative / positive).sqrt().clamp(1.0, 20.0)
    valid = footprint.bool()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        expected,
        pos_weight=positive_weight,
        reduction="none",
    )
    return bce[valid].mean() + dice_loss(logits, expected)


def structured_object_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    opening_weight: float = 1.0,
    furniture_weight: float = 1.0,
    wall_weight: float = 1.0,
    positive_class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    opening_ids = torch.tensor(taxonomy.opening_ids, device=target.device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=target.device)
    opening_weights = None
    furniture_weights = None
    if positive_class_weights is not None:
        weights = positive_class_weights.to(target.device)
        opening_weights = weights[opening_ids - 1]
        furniture_weights = weights[furniture_ids - 1]
    opening_loss, opening_ce, opening_dice = group_ce_dice_loss(
        outputs["opening_logits"],
        target,
        opening_ids,
        footprint,
        opening_weights,
        outputs.get("opening_background_logits"),
    )
    furniture_loss, furniture_ce, furniture_dice = group_ce_dice_loss(
        outputs["furniture_logits"],
        target,
        furniture_ids,
        footprint,
        furniture_weights,
        outputs.get("furniture_background_logits"),
    )
    wall_loss, _wall_bce, _wall_tversky = interior_wall_loss(
        outputs["wall_logits"],
        target,
        footprint,
        taxonomy,
        target_dilation=0,
    )
    total = opening_weight * opening_loss + furniture_weight * furniture_loss + wall_weight * wall_loss
    return total, {
        "structured_object_loss": total.detach(),
        "structured_opening_loss": opening_loss.detach(),
        "structured_opening_ce": opening_ce,
        "structured_opening_dice": opening_dice,
        "structured_furniture_loss": furniture_loss.detach(),
        "structured_furniture_ce": furniture_ce,
        "structured_furniture_dice": furniture_dice,
        "structured_wall_loss": wall_loss.detach(),
    }


@torch.no_grad()
def object_group_predictions(
    outputs: dict[str, torch.Tensor],
    taxonomy: FloorplanTaxonomy,
    *,
    opening_threshold: float | torch.Tensor = 0.0,
    furniture_threshold: float | torch.Tensor = 0.0,
    wall_threshold: float = 0.0,
    score_scale: str = "logit",
) -> dict[str, torch.Tensor]:
    opening_ids = torch.tensor(taxonomy.opening_ids, device=outputs["opening_logits"].device)
    furniture_ids = torch.tensor(taxonomy.furniture_ids, device=outputs["furniture_logits"].device)
    opening_scores = outputs["opening_logits"]
    furniture_scores = outputs["furniture_logits"]
    opening_index = opening_scores.argmax(dim=1)
    furniture_index = furniture_scores.argmax(dim=1)
    if score_scale == "probability":
        opening_background = outputs.get(
            "opening_background_logits",
            torch.zeros_like(opening_scores[:, 0]),
        )
        furniture_background = outputs.get(
            "furniture_background_logits",
            torch.zeros_like(furniture_scores[:, 0]),
        )
        opening_probability = torch.softmax(
            torch.cat([opening_background[:, None], opening_scores], dim=1),
            dim=1,
        )
        furniture_probability = torch.softmax(
            torch.cat([furniture_background[:, None], furniture_scores], dim=1),
            dim=1,
        )
        opening_confidence = opening_probability[:, 1:].max(dim=1).values
        furniture_confidence = furniture_probability[:, 1:].max(dim=1).values
    else:
        opening_confidence = opening_scores.max(dim=1).values
        furniture_confidence = furniture_scores.max(dim=1).values
    def decision_threshold(threshold: float | torch.Tensor, class_index: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(threshold):
            return threshold.to(device=class_index.device, dtype=opening_scores.dtype)[class_index]
        return torch.full_like(class_index, float(threshold), dtype=opening_scores.dtype)

    return {
        "opening": torch.where(
            opening_confidence > decision_threshold(opening_threshold, opening_index),
            opening_ids[opening_index],
            torch.zeros_like(opening_index, dtype=opening_ids.dtype),
        ),
        "furniture": torch.where(
            furniture_confidence > decision_threshold(furniture_threshold, furniture_index),
            furniture_ids[furniture_index],
            torch.zeros_like(furniture_index, dtype=furniture_ids.dtype),
        ),
        "wall": (outputs["wall_logits"] > wall_threshold).long(),
    }


def compose_structured_objects(
    room_labels: torch.Tensor,
    object_predictions: dict[str, torch.Tensor],
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    labels = room_labels.clone()
    wall_id = taxonomy.class_id("wall")
    furniture = object_predictions["furniture"]
    labels = torch.where(furniture != 0, furniture, labels)
    labels = torch.where(object_predictions["wall"].bool(), torch.full_like(labels, wall_id), labels)
    opening = object_predictions["opening"]
    labels = torch.where(opening != 0, opening, labels)
    return labels * footprint.long()


def compatible_furniture_mask(
    furniture_labels: torch.Tensor,
    room_labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> torch.Tensor:
    """Keep prior furniture only if it lands in a semantically plausible room."""

    output = torch.zeros_like(furniture_labels, dtype=torch.bool)
    groups = (
        (
            (
                "Toilet",
                "Bathtub",
                "Sink",
                "RoundSink",
                "DoubleSink",
                "DoubleSinkRight",
                "SideSink",
                "Shower",
                "ShowerCab",
                "ShowerScreen",
            ),
            ("Bath", "Sauna", "Utility"),
        ),
        (("BaseCabinet", "WallCabinet", "ElectricalAppliance"), ("Kitchen", "Utility")),
        (("Fireplace", "FireplaceCorner", "WoodStove"), ("LivingRoom", "Den")),
        (("Closet", "CoatCloset", "CoatRack"), ("Closet", "DressingRoom", "Storage", "Entry", "Hall")),
        (("SaunaBench", "SaunaBenchHigh", "SaunaBenchLow", "SaunaBenchMid"), ("Sauna", "Bath")),
    )
    for furniture_names, room_names in groups:
        furniture_ids = [taxonomy.class_id(f"furniture:{name}") for name in furniture_names]
        room_ids = [taxonomy.class_id(f"room:{name}") for name in room_names]
        in_group = torch.isin(furniture_labels, torch.tensor(furniture_ids, device=furniture_labels.device))
        in_room = torch.isin(room_labels, torch.tensor(room_ids, device=room_labels.device))
        output |= in_group & in_room
    return output


def partition_wall_mask(room_labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Derive thin walls from changes between adjacent generated room types."""

    horizontal = torch.zeros_like(room_labels, dtype=torch.bool)
    vertical = torch.zeros_like(room_labels, dtype=torch.bool)
    horizontal[:, :, 1:] = room_labels[:, :, 1:] != room_labels[:, :, :-1]
    vertical[:, 1:, :] = room_labels[:, 1:, :] != room_labels[:, :-1, :]
    boundary = (horizontal | vertical).float()[:, None]
    if radius > 0:
        boundary = F.max_pool2d(boundary, 2 * radius + 1, stride=1, padding=radius)
    return boundary[:, 0] > 0.5


def constrain_structured_objects(
    predictions: dict[str, torch.Tensor],
    room_labels: torch.Tensor,
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    *,
    wall_radius: int = 0,
    support_radius: int = 2,
    use_predicted_walls: bool = True,
    require_compatible_furniture: bool = True,
) -> dict[str, torch.Tensor]:
    """Project independent object heads onto room-partition geometry.

    Walls may only occur near room transitions, openings need wall support, and
    known furniture families must be compatible with their predicted room type.
    Unknown or unsupported pixels are removed rather than allowed to overwrite
    the room partition.
    """

    rooms = room_labels * footprint.long()
    boundary = partition_wall_mask(rooms, radius=max(0, int(wall_radius)))
    wall_support = F.max_pool2d(
        boundary.float()[:, None],
        2 * int(support_radius) + 1,
        stride=1,
        padding=int(support_radius),
    )[:, 0] > 0.5
    walls = boundary
    if use_predicted_walls:
        walls = walls | (predictions["wall"].bool() & wall_support)
    wall_dilation = F.max_pool2d(
        walls.float()[:, None],
        2 * int(wall_radius) + 1,
        stride=1,
        padding=int(wall_radius),
    )[:, 0] > 0.5
    openings = torch.where(
        wall_dilation,
        predictions["opening"],
        torch.zeros_like(predictions["opening"]),
    )
    if require_compatible_furniture:
        furniture = torch.where(
            compatible_furniture_mask(predictions["furniture"], rooms, taxonomy),
            predictions["furniture"],
            torch.zeros_like(predictions["furniture"]),
        )
    else:
        furniture = predictions["furniture"]
    return {
        "wall": walls.long(),
        "opening": openings,
        "furniture": furniture,
    }


@torch.no_grad()
def object_group_metrics(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> dict[str, float]:
    metrics = {}
    for name, class_ids in (("opening", taxonomy.opening_ids), ("furniture", taxonomy.furniture_ids)):
        predicted_mask = torch.isin(predictions[name], torch.tensor(class_ids, device=target.device))
        expected_mask = torch.isin(target, torch.tensor(class_ids, device=target.device))
        intersection = (predicted_mask & expected_mask).sum().float()
        union = (predicted_mask | expected_mask).sum().float()
        metrics[f"{name}_iou"] = float(intersection / union.clamp_min(1.0))
        precision = intersection / predicted_mask.sum().clamp_min(1).float()
        recall = intersection / expected_mask.sum().clamp_min(1).float()
        metrics[f"{name}_f1"] = float(2.0 * precision * recall / (precision + recall).clamp_min(1.0e-6))
    wall_id = taxonomy.class_id("wall")
    predicted_wall = predictions["wall"].bool()
    expected_wall = target == wall_id
    metrics["wall_iou"] = float(
        (predicted_wall & expected_wall).sum().float()
        / (predicted_wall | expected_wall).sum().clamp_min(1).float()
    )
    return metrics
