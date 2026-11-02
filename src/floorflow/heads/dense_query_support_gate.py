"""Learned per-pixel arbitration between dense and query room evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.ndimage import label as component_label
from torch import nn

from floorflow.heads.room_segmenter import ConvNormAct


def query_mask_support_probability(
    mask_logits: torch.Tensor,
    *,
    mode: str = "union",
    existence_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert query masks into an explicit room-support probability."""

    probability = mask_logits.sigmoid()
    if existence_logits is not None:
        if existence_logits.ndim != 2:
            raise ValueError("existence logits must be [B,num_queries]")
        if existence_logits.shape != probability.shape[:2]:
            raise ValueError("existence and mask queries must align")
        probability = probability * existence_logits.sigmoid()[..., None, None]
    if mode == "max":
        return probability.amax(dim=1)
    if mode == "union":
        return 1.0 - (1.0 - probability).prod(dim=1)
    raise ValueError(f"unsupported query support mode: {mode}")


def apply_room_support_gate(
    labels: torch.Tensor,
    support_probability: torch.Tensor,
    footprint: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Attach a consumed support mask to room labels."""

    if labels.ndim != 3 or support_probability.ndim != 3:
        raise ValueError("labels and support must be [B,H,W]")
    if labels.shape != support_probability.shape:
        raise ValueError("labels and support must share a shape")
    if footprint.ndim == 4:
        if footprint.shape[1] != 1:
            raise ValueError("footprint must have one channel")
        footprint = footprint[:, 0]
    if footprint.shape != labels.shape:
        raise ValueError("footprint must match labels")
    keep = (
        support_probability >= float(threshold)
    ) & footprint.bool()
    return labels.long() * keep.long()


def topology_safe_keep_mask(
    labels: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    minimum_deletable_component_area: int = 32,
) -> torch.Tensor:
    """Reject proposed deletions that split any room connected component."""

    if labels.shape != keep_mask.shape:
        raise ValueError("labels and keep mask must share a shape")
    squeeze = labels.ndim == 2
    if squeeze:
        labels = labels[None]
        keep_mask = keep_mask[None]
    if labels.ndim != 3:
        raise ValueError("labels and keep mask must be [B,H,W] or [H,W]")
    structure = torch.tensor(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=torch.uint8,
    ).numpy()
    output = keep_mask.bool().clone()
    labels_np = labels.detach().cpu().numpy()
    keep_np = output.detach().cpu().numpy()
    for batch in range(labels.shape[0]):
        for class_id in torch.unique(labels[batch]).tolist():
            if int(class_id) <= 0:
                continue
            components, count = component_label(
                labels_np[batch] == int(class_id),
                structure=structure,
            )
            for component_id in range(1, int(count) + 1):
                component = components == component_id
                remaining = component & keep_np[batch]
                if not remaining.any():
                    if component.sum() >= int(
                        minimum_deletable_component_area
                    ):
                        output[batch, component] = True
                    continue
                _remaining_components, remaining_count = component_label(
                    remaining,
                    structure=structure,
                )
                if remaining_count > 1:
                    output[batch, component] = True
    return output[0] if squeeze else output


def render_support_conditioned_room_labels(
    probability: torch.Tensor,
    keep_mask: torch.Tensor,
    room_ids: torch.Tensor,
    footprint: torch.Tensor,
    *,
    topology_safe: bool,
    minimum_deletable_component_area: int = 32,
) -> torch.Tensor:
    """Render support labels in the order required by the metric consumer."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    footprint = footprint.bool()
    labels = room_ids.to(probability.device)[
        probability.argmax(dim=1)
    ] * footprint.long()
    keep = keep_mask.bool() & footprint
    if topology_safe:
        keep = topology_safe_keep_mask(
            labels,
            keep,
            minimum_deletable_component_area=(
                minimum_deletable_component_area
            ),
        )
    return labels * keep.long()


def dense_query_gate_features(
    dense_probability: torch.Tensor,
    query_probability: torch.Tensor,
    query_support: torch.Tensor,
    footprint: torch.Tensor,
) -> torch.Tensor:
    """Build class-sensitive and confidence-sensitive gate features."""

    if dense_probability.shape != query_probability.shape:
        raise ValueError("dense and query probabilities must share a shape")
    if footprint.ndim == 3:
        footprint = footprint[:, None].bool()
    query_support = query_support[:, None].float()
    dense_top = dense_probability.topk(2, dim=1)
    query_top = query_probability.topk(2, dim=1)
    dense_entropy = -(
        dense_probability.clamp_min(1.0e-6)
        * dense_probability.clamp_min(1.0e-6).log()
    ).sum(dim=1, keepdim=True)
    query_entropy = -(
        query_probability.clamp_min(1.0e-6)
        * query_probability.clamp_min(1.0e-6).log()
    ).sum(dim=1, keepdim=True)
    agreement = (
        dense_probability.argmax(dim=1, keepdim=True)
        == query_probability.argmax(dim=1, keepdim=True)
    ).float()
    return torch.cat(
        [
            dense_probability,
            query_probability,
            torch.abs(dense_probability - query_probability),
            dense_top.values[:, :1],
            (dense_top.values[:, :1] - dense_top.values[:, 1:2]),
            query_top.values[:, :1],
            (query_top.values[:, :1] - query_top.values[:, 1:2]),
            query_support,
            dense_entropy,
            query_entropy,
            agreement,
            footprint.float(),
        ],
        dim=1,
    )


class DenseQuerySupportGate(nn.Module):
    """Predict a residual query weight over a fixed dense/query blend."""

    def __init__(
        self,
        num_room_classes: int,
        *,
        fixed_blend: float = 0.5,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(fixed_blend) <= 1.0:
            raise ValueError("fixed_blend must be in [0, 1]")
        self.fixed_blend = float(fixed_blend)
        input_channels = 3 * int(num_room_classes) + 9
        hidden = int(hidden_channels)
        self.network = nn.Sequential(
            ConvNormAct(input_channels, hidden),
            ConvNormAct(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        dense_probability: torch.Tensor,
        query_probability: torch.Tensor,
        query_support: torch.Tensor,
        footprint: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = dense_query_gate_features(
            dense_probability,
            query_probability,
            query_support,
            footprint,
        )
        residual_logits = self.network(features)
        residual = self.residual_scale.tanh() * torch.tanh(residual_logits)
        gate = (self.fixed_blend + residual).clamp(0.0, 1.0)
        blended = (
            (1.0 - gate) * dense_probability
            + gate * query_probability
        )
        return {
            "gate": gate[:, 0],
            "gate_residual_logits": residual_logits[:, 0],
            "probability": blended,
        }


class SelectiveRoomSupportDeletion(nn.Module):
    """Delete low-confidence room support while preserving agreed room cores."""

    def __init__(
        self,
        num_room_classes: int,
        *,
        fixed_blend: float = 0.5,
        hidden_channels: int = 16,
        preserve_confidence: float = 0.8,
    ) -> None:
        super().__init__()
        self.fixed_blend = float(fixed_blend)
        self.preserve_confidence = float(preserve_confidence)
        input_channels = 3 * int(num_room_classes) + 9
        hidden = int(hidden_channels)
        self.network = nn.Sequential(
            ConvNormAct(input_channels, hidden),
            ConvNormAct(hidden, hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        dense_probability: torch.Tensor,
        query_probability: torch.Tensor,
        query_support: torch.Tensor,
        footprint: torch.Tensor,
        *,
        keep_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if footprint.ndim == 3:
            footprint = footprint[:, None].bool()
        if footprint.shape[1] != 1:
            raise ValueError("footprint must have one channel")
        footprint_2d = footprint[:, 0]
        features = dense_query_gate_features(
            dense_probability,
            query_probability,
            query_support,
            footprint,
        )
        blended = (
            (1.0 - self.fixed_blend) * dense_probability
            + self.fixed_blend * query_probability
        )
        deletion_logits = self.network(features)[:, 0]
        deletion_probability = deletion_logits.sigmoid()
        dense_top = dense_probability.argmax(dim=1)
        query_top = query_probability.argmax(dim=1)
        blend_top = blended.argmax(dim=1)
        confidence = torch.maximum(
            dense_probability.amax(dim=1),
            query_probability.amax(dim=1),
        )
        preserve = (
            footprint_2d
            & (dense_top == query_top)
            & (dense_top == blend_top)
            & (confidence >= self.preserve_confidence)
        )
        keep = footprint_2d & (
            preserve | (deletion_probability < float(keep_threshold))
        )
        return {
            "probability": blended,
            "deletion_logits": deletion_logits,
            "deletion_probability": deletion_probability,
            "hard_preserve": preserve,
            "keep_mask": keep,
        }


def selective_deletion_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    *,
    false_deletion_weight: float = 5.0,
    deletion_weight: float = 1.0,
    support_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train deletion only on non-room support with asymmetric room weights."""

    if footprint.ndim == 4:
        footprint = footprint[:, 0]
    footprint = footprint.bool()
    room_target = torch.isin(target, target.new_tensor(room_ids))
    drop_target = footprint & ~room_target
    weights = torch.where(
        room_target,
        room_target.new_tensor(float(false_deletion_weight)),
        room_target.new_tensor(1.0),
    )
    deletion_loss = F.binary_cross_entropy_with_logits(
        outputs["deletion_logits"][footprint],
        drop_target[footprint].float(),
        weight=weights[footprint],
    )
    keep_probability = 1.0 - outputs["deletion_probability"]
    support_loss = F.binary_cross_entropy(
        keep_probability[footprint].clamp(1.0e-5, 1.0 - 1.0e-5),
        room_target[footprint].float(),
    )
    loss = (
        float(deletion_weight) * deletion_loss
        + float(support_weight) * support_loss
    )
    predicted_drop = outputs["deletion_logits"].sigmoid() >= 0.5
    return loss, {
        "selective_deletion_loss": loss.detach(),
        "selective_deletion_bce": deletion_loss.detach(),
        "selective_support_loss": support_loss.detach(),
        "selective_false_deletion_rate": (
            (predicted_drop & room_target & footprint).sum()
            / room_target[footprint].sum().clamp_min(1)
        ).detach(),
    }


def support_gate_targets(
    target: torch.Tensor,
    dense_probability: torch.Tensor,
    query_probability: torch.Tensor,
    room_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target room support and branch-arbitration labels."""

    room_tensor = target.new_tensor(room_ids)
    room_target = torch.isin(target, room_tensor)
    dense_top = dense_probability.argmax(dim=1)
    query_top = query_probability.argmax(dim=1)
    target_lookup = torch.full_like(target, -1)
    for index, class_id in enumerate(room_ids):
        target_lookup[target == int(class_id)] = index
    valid = room_target & (target_lookup >= 0)
    dense_correct = valid & (dense_top == target_lookup)
    query_correct = valid & (query_top == target_lookup)
    arbitration = torch.full_like(target, -1)
    arbitration[query_correct & ~dense_correct] = 1
    arbitration[dense_correct & ~query_correct] = 0
    return room_target, arbitration


def support_gate_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    dense_probability: torch.Tensor,
    query_probability: torch.Tensor,
    footprint: torch.Tensor,
    room_ids: tuple[int, ...],
    *,
    arbitration_weight: float = 0.25,
    support_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise blended room support and branch choice on disagreement."""

    if footprint.ndim == 3:
        footprint = footprint[:, None].bool()
    room_target, arbitration = support_gate_targets(
        target,
        dense_probability,
        query_probability,
        room_ids,
    )
    support = outputs["probability"].amax(dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
    support_loss = F.binary_cross_entropy(
        support[footprint[:, 0]],
        room_target[footprint[:, 0]].float(),
    )
    valid_arbitration = arbitration >= 0
    arbitration_loss = outputs["gate"].sum() * 0.0
    if valid_arbitration.any():
        arbitration_loss = F.binary_cross_entropy(
            outputs["gate"][valid_arbitration].clamp(0.0, 1.0),
            arbitration[valid_arbitration].float(),
        )
    total = (
        float(support_weight) * support_loss
        + float(arbitration_weight) * arbitration_loss
    )
    return total, {
        "support_gate_loss": total.detach(),
        "support_gate_support_loss": support_loss.detach(),
        "support_gate_arbitration_loss": arbitration_loss.detach(),
        "support_gate_arbitration_pixels": valid_arbitration.float().mean().detach(),
    }


__all__ = [
    "DenseQuerySupportGate",
    "SelectiveRoomSupportDeletion",
    "apply_room_support_gate",
    "dense_query_gate_features",
    "query_mask_support_probability",
    "render_support_conditioned_room_labels",
    "selective_deletion_loss",
    "support_gate_loss",
    "support_gate_targets",
    "topology_safe_keep_mask",
]
