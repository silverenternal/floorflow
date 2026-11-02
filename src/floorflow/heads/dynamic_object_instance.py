"""Dynamic object-query heads adapted from Mask2Former and PMT patterns."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.ndimage import label
from torch import nn

from floorflow.heads.opening_interval_head import WallIntervalHead
from floorflow.heads.room_mask_decoder import RoomMaskDecoder
from floorflow.objectives.furniture_vector_refine import soft_object_render


def object_instance_targets(
    labels: torch.Tensor,
    class_ids: tuple[int, ...],
    max_instances: int = 24,
    min_area: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract class-wise connected-component masks for object queries."""

    height, width = labels.shape[-2:]
    batch_classes = []
    batch_masks = []
    batch_valid = []
    for item in labels.detach().cpu().numpy():
        instances = []
        for class_index, class_id in enumerate(class_ids):
            components, count = label(item == int(class_id))
            for component_id in range(1, int(count) + 1):
                mask = components == component_id
                area = int(mask.sum())
                if area < int(min_area):
                    continue
                instances.append((area, class_index, mask))
        instances.sort(key=lambda value: value[0], reverse=True)
        instances = instances[: int(max_instances)]
        classes = torch.zeros(max_instances, dtype=torch.long)
        masks = torch.zeros(max_instances, height, width)
        valid = torch.zeros(max_instances, dtype=torch.bool)
        for index, (_area, class_index, mask) in enumerate(instances):
            classes[index] = class_index
            masks[index] = torch.from_numpy(mask.astype("float32"))
            valid[index] = True
        batch_classes.append(classes)
        batch_masks.append(masks)
        batch_valid.append(valid)
    return (
        torch.stack(batch_classes).to(labels.device),
        torch.stack(batch_masks).to(labels.device),
        torch.stack(batch_valid).to(labels.device),
    )


class DynamicFurnitureInstanceHead(nn.Module):
    """Room-conditioned furniture instance queries with a dynamic query budget."""

    def __init__(
        self,
        in_channels: int,
        num_furniture_classes: int,
        *,
        hidden_dim: int = 96,
        max_queries: int = 24,
        min_queries: int = 4,
    ) -> None:
        super().__init__()
        if min_queries <= 0 or max_queries < min_queries:
            raise ValueError("invalid furniture query bounds")
        self.min_queries = int(min_queries)
        self.max_queries = int(max_queries)
        self.num_furniture_classes = int(num_furniture_classes)
        self.room_fusion = nn.Conv2d(1, in_channels, 1)
        nn.init.normal_(
            self.room_fusion.weight,
            mean=0.0,
            std=1.0e-3,
        )
        nn.init.zeros_(self.room_fusion.bias)
        self.decoder = RoomMaskDecoder(
            in_channels,
            int(num_furniture_classes),
            hidden_dim=hidden_dim,
            num_queries=max_queries,
            layers=2,
            heads=4,
            feature_scales=(4, 8),
        )

    def _query_budget(
        self,
        room_probability: torch.Tensor,
        footprint: torch.Tensor,
        class_density: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = footprint.float().flatten(1).sum(dim=1).clamp_min(1.0)
        room_mass = room_probability.float().flatten(1).sum(dim=1)
        complexity = (room_mass / valid).clamp(0.0, 1.0)
        budget = torch.ceil(
            self.min_queries
            + complexity * (self.max_queries - self.min_queries)
        )
        if class_density is not None:
            density = class_density.float().to(budget.device).clamp_min(0.0)
            class_mass = density.sum(dim=(2, 3))
            mass = class_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            concentration = (class_mass / mass).amax(dim=1).clamp(0.0, 1.0)
            budget = budget + 0.5 * concentration * (
                self.max_queries - self.min_queries
            )
        return budget.clamp(
            float(self.min_queries),
            float(self.max_queries),
        ).long()

    def forward(
        self,
        features: torch.Tensor,
        room_probability: torch.Tensor,
        footprint: torch.Tensor,
        class_density: torch.Tensor | None = None,
        room_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        if class_density is not None and (
            class_density.ndim != 4
            or class_density.shape[0] != features.shape[0]
            or class_density.shape[1] != self.num_furniture_classes
        ):
            raise ValueError(
                "class_density must be [B,num_furniture_classes,H,W]"
            )
        if room_probability.shape[-2:] != features.shape[-2:]:
            room_probability = F.interpolate(
                room_probability.float(),
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        conditioned = features.float() + self.room_fusion(
            room_probability.clamp(0.0, 1.0)
        )
        decoded = self.decoder(
            conditioned,
            prototype_logits=class_density,
            room_tokens=room_tokens,
            include_prototype_memory=False,
        )
        decoded["mask_logits"] = F.interpolate(
            decoded["mask_logits"],
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded["aux_mask_logits"] = [
            F.interpolate(
                value,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for value in decoded["aux_mask_logits"]
        ]
        budget = self._query_budget(
            room_probability,
            footprint,
            class_density=class_density,
        )
        class_confidence = decoded["class_logits"][..., :-1].softmax(
            dim=-1
        ).amax(dim=-1)
        mask_peak = decoded["mask_logits"].sigmoid().flatten(2).amax(dim=-1)
        query_score = class_confidence * mask_peak
        order = query_score.argsort(dim=-1, descending=True, stable=True)
        positions = torch.arange(
            self.max_queries,
            device=features.device,
        )[None]
        active = torch.zeros_like(budget[:, None], dtype=torch.bool).expand(
            -1,
            self.max_queries,
        )
        active = active.scatter(
            1,
            order,
            positions < budget[:, None],
        )
        decoded["active_queries"] = active
        decoded["query_budget"] = budget
        return decoded


def wall_endpoints_from_masks(
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert one wall mask per token into endpoint pairs."""

    if masks.ndim != 4:
        raise ValueError("wall masks must be [B,W,H,W]")
    endpoints = []
    for sample in masks:
        sample_rows = []
        for mask in sample:
            index = (mask.sigmoid() >= float(threshold)).nonzero(
                as_tuple=False
            )
            if index.numel() == 0:
                sample_rows.append(mask.new_zeros(4))
                continue
            y0 = index[:, 0].min()
            y1 = index[:, 0].max()
            x0 = index[:, 1].min()
            x1 = index[:, 1].max()
            sample_rows.append(
                torch.stack([x0, y0, x1, y1]).to(mask.dtype)
            )
        endpoints.append(torch.stack(sample_rows))
    return torch.stack(endpoints)


def render_object_queries(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    class_ids: tuple[int, ...],
    footprint: torch.Tensor,
    *,
    presence_threshold: float = 0.2,
    mask_threshold: float = 0.4,
    active_queries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render non-overlapping object instances from query scores."""

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if footprint.ndim != 4 or footprint.shape[1] != 1:
        raise ValueError("footprint must be [B,H,W] or [B,1,H,W]")
    if mask_logits.shape[-2:] != footprint.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=footprint.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    probability = class_logits.softmax(dim=-1)[..., :-1]
    best_probability, best_class = probability.max(dim=-1)
    mask_probability = mask_logits.sigmoid()
    score = mask_probability * best_probability[..., None, None]
    if active_queries is not None:
        if active_queries.shape != best_probability.shape:
            raise ValueError("active_queries must match [B,Q]")
        if not bool(active_queries.any(dim=1).all()):
            raise ValueError("each sample must retain one active query")
        score = score.masked_fill(
            ~active_queries.bool()[..., None, None],
            -1.0,
        )
    flat_score = score.flatten(2)
    selected_mask, selected_query = flat_score.max(dim=1)
    selected_mask = selected_mask.reshape(score.shape[0], *score.shape[-2:])
    selected_class = torch.gather(best_class, 1, selected_query)
    selected_presence = torch.gather(
        best_probability,
        1,
        selected_query,
    ).reshape(score.shape[0], *score.shape[-2:])
    class_ids_tensor = torch.tensor(
        class_ids,
        device=class_logits.device,
        dtype=torch.long,
    )
    labels = class_ids_tensor[selected_class].reshape(
        score.shape[0],
        *score.shape[-2:],
    )
    valid = (
        (selected_mask >= float(mask_threshold))
        & (selected_presence >= float(presence_threshold))
        & footprint[:, 0].bool()
    )
    return labels * valid.long()


def render_object_queries_soft(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    footprint: torch.Tensor,
    *,
    temperature: float = 1.0,
    active_queries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable training-time counterpart of :func:`render_object_queries`."""

    return soft_object_render(
        class_logits,
        mask_logits,
        footprint,
        temperature=temperature,
        active_queries=active_queries,
    )


class DynamicWallOpeningIntervalHead(nn.Module):
    """Use PMT-style wall/opening tokens to predict attached intervals."""

    def __init__(
        self,
        hidden_dim: int,
        num_opening_types: int,
        *,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.interval = WallIntervalHead(
            hidden_dim,
            num_types=num_opening_types,
            heads=heads,
        )
        self.presence_gate = nn.Linear(hidden_dim * 2, 1)
        nn.init.zeros_(self.presence_gate.weight)
        nn.init.zeros_(self.presence_gate.bias)

    def forward(
        self,
        wall_tokens: torch.Tensor,
        opening_tokens: torch.Tensor,
        wall_masks: torch.Tensor,
        *,
        wall_mask: torch.Tensor | None = None,
        opening_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        endpoints = wall_endpoints_from_masks(wall_masks)
        result = self.interval(
            wall_tokens,
            opening_tokens,
            endpoints,
            wall_mask=wall_mask,
            opening_mask=opening_mask,
        )
        summary = opening_tokens.mean(dim=1, keepdim=True).expand(
            -1,
            wall_tokens.shape[1],
            -1,
        )
        alignment_gate = torch.sigmoid(
            self.presence_gate(torch.cat([wall_tokens, summary], dim=-1))
        )[..., 0]
        result["presence_logits"] = (
            result["presence_logits"] + alignment_gate.clamp_min(1.0e-4).log()
        )
        result["wall_endpoints"] = endpoints
        return result


__all__ = [
    "DynamicFurnitureInstanceHead",
    "DynamicWallOpeningIntervalHead",
    "object_instance_targets",
    "render_object_queries",
    "wall_endpoints_from_masks",
]
