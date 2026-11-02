"""Mask2Former-inspired query decoder for room instances.

The masked cross-attention and query-mask design is adapted from
facebookresearch/Mask2Former (commit 9b0651c, MIT license):
https://github.com/facebookresearch/Mask2Former
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.heads.adaptive_query_reference import (
    AdaptiveQueryReferenceRefiner,
)
from floorflow.heads.attention_masks import (
    soft_owner_attention_bias,
    soft_pixel_attention_bias,
    soft_presence_attention_bias,
    unmask_first_empty_row,
)
from floorflow.heads.room_mask_class_fusion import RoomMaskClassFusion
from floorflow.heads.room_mask_fusion import fused_room_mask_logits, render_room_masks
from floorflow.heads.room_mask_losses import pairwise_mask_cost as _pairwise_mask_cost
from floorflow.heads.room_mask_losses import room_mask_loss, room_mask_partition_loss
from floorflow.heads.room_partition import batch_room_partition_target
from floorflow.representation.core import extract_room_instances

__all__ = [
    "RoomMaskDecoder",
    "fused_room_mask_logits",
    "query_existence_from_class_logits",
    "room_mask_loss",
    "room_mask_metrics",
    "room_mask_partition_loss",
    "room_mask_targets",
]


@lru_cache(maxsize=16)
def _cached_position_embedding(
    height: int,
    width: int,
    channels: int,
    device: str,
) -> torch.Tensor:
    """Fixed sinusoidal x/y position embedding for room queries."""

    if channels % 4:
        raise ValueError("position embedding channels must be divisible by 4")
    resolved = torch.device(device)
    frequencies = torch.exp(
        -np.log(10_000.0)
        * torch.arange(
            channels // 4,
            device=resolved,
            dtype=torch.float32,
        )
        / max(1, channels // 4 - 1)
    )
    y = torch.linspace(0.0, 1.0, height, device=resolved)
    x = torch.linspace(0.0, 1.0, width, device=resolved)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x_encoding = torch.cat(
        [torch.sin(xx[..., None] * frequencies), torch.cos(xx[..., None] * frequencies)],
        dim=-1,
    )
    y_encoding = torch.cat(
        [torch.sin(yy[..., None] * frequencies), torch.cos(yy[..., None] * frequencies)],
        dim=-1,
    )
    return torch.cat([x_encoding, y_encoding], dim=-1).flatten(0, 1)[None]


def position_embedding_2d(
    height: int,
    width: int,
    channels: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached fixed sinusoidal position embedding."""

    return _cached_position_embedding(
        int(height),
        int(width),
        int(channels),
        str(device),
    )


class RoomMaskDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.heads = int(heads)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        normalized = self.cross_norm(queries)
        batch, num_queries, _ = queries.shape
        repeated_mask = attention_mask[:, None].expand(-1, self.heads, -1, -1)
        repeated_mask = repeated_mask.reshape(batch * self.heads, num_queries, memory.shape[1])
        attended = self.cross_attention(
            normalized,
            memory,
            memory,
            attn_mask=repeated_mask,
            need_weights=False,
        )[0]
        queries = queries + attended
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class RoomMaskDecoder(nn.Module):
    """Predict room masks and classes as a set, without box quantization."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 128,
        num_queries: int = 32,
        layers: int = 3,
        heads: int = 4,
        feature_scales: tuple[int, ...] = (1, 2, 4),
        program_token_dim: int = 0,
        partition_consistent: bool = False,
        partition_attention_masking: bool = True,
        adaptive_query_refinement: bool = False,
        adaptive_query_residual_scale: float = 0.0,
        full_resolution_mask_refinement: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.partition_consistent = bool(partition_consistent)
        self.partition_attention_masking = bool(partition_attention_masking)
        self.adaptive_query_refinement = bool(adaptive_query_refinement)
        self.full_resolution_mask_refinement = bool(
            full_resolution_mask_refinement
        )
        self.mask_features = nn.Conv2d(in_channels, hidden_dim, 1)
        self.prototype_projection = nn.Conv2d(num_room_classes, hidden_dim, 1)
        self.feature_scales = tuple(sorted({max(1, int(scale)) for scale in feature_scales}))
        self.query_feature = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)
        self.adaptive_query_refiner = (
            AdaptiveQueryReferenceRefiner(
                hidden_dim,
                in_channels,
                live_reference_residual_scale=(
                    adaptive_query_residual_scale
                ),
            )
            if self.adaptive_query_refinement
            else None
        )
        self.fusion_scale = nn.Parameter(torch.zeros(1))
        self.full_resolution_mask_scale = (
            nn.Parameter(torch.tensor(0.1))
            if self.full_resolution_mask_refinement
            else None
        )
        self.layers = nn.ModuleList([RoomMaskDecoderLayer(hidden_dim, heads) for _ in range(layers)])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_fusion = RoomMaskClassFusion(
            hidden_dim,
            heads,
            program_token_dim=program_token_dim,
        )
        self.class_head = nn.Linear(hidden_dim, num_room_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.mask_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.mask_head[-1].bias)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        prototype_logits: torch.Tensor | None = None,
        program_tokens: torch.Tensor | None = None,
        room_tokens: torch.Tensor | None = None,
        room_token_valid: torch.Tensor | None = None,
        adaptive_features: torch.Tensor | None = None,
        full_resolution_prototype_logits: torch.Tensor | None = None,
        include_prototype_memory: bool = True,
    ) -> dict[str, torch.Tensor]:
        height, width = features.shape[-2:]
        levels = []
        projected_full = self.mask_features(features)
        for scale in self.feature_scales:
            projected = (
                projected_full
                if scale == 1
                else F.avg_pool2d(
                    projected_full,
                    scale,
                    stride=scale,
                )
            )
            memory = projected.flatten(2).transpose(1, 2)
            memory = memory + position_embedding_2d(
                projected.shape[-2],
                projected.shape[-1],
                memory.shape[-1],
                memory.device,
            ).to(memory.dtype)
            levels.append((projected, memory))
        memory = torch.cat([item[1] for item in levels], dim=1)
        prototype_tokens = None
        if prototype_logits is not None:
            prototypes = F.interpolate(
                prototype_logits.float(),
                size=(height, width),
                mode="nearest",
            )
            projected_prototypes = self.prototype_projection(
                prototypes
            ).flatten(2).transpose(1, 2)
            prototype_tokens = projected_prototypes
            prototype_tokens = prototype_tokens + position_embedding_2d(
                height,
                width,
                memory.shape[-1],
                memory.device,
            ).to(memory.dtype)
            if include_prototype_memory:
                memory = torch.cat([memory, prototype_tokens], dim=1)
        room_token_padding = None
        if room_tokens is not None:
            if room_tokens.ndim != 3 or room_tokens.shape[-1] != memory.shape[-1]:
                raise ValueError("room_tokens must be [B,R,H] with decoder width")
            if room_token_valid is not None:
                if room_token_valid.shape != room_tokens.shape[:2]:
                    raise ValueError(
                        "room_token_valid must be [B,R] matching room_tokens"
                    )
                room_tokens = torch.where(
                    room_token_valid[..., None],
                    room_tokens,
                    torch.zeros_like(room_tokens),
                )
                room_token_padding = ~room_token_valid.bool()
                room_token_padding = unmask_first_empty_row(
                    room_token_padding
                )
            memory = torch.cat([memory, room_tokens], dim=1)
        queries = self.query_feature.weight[None].expand(features.shape[0], -1, -1)
        queries = queries + self.query_position.weight[None]
        adaptive_diagnostics = None
        if self.adaptive_query_refiner is not None:
            sampled_features = (
                features
                if adaptive_features is None
                else adaptive_features
            )
            query_update, adaptive_diagnostics = self.adaptive_query_refiner(
                queries,
                sampled_features,
                reference_logits=prototype_logits,
            )
            queries = queries + query_update
        mask_logits = None
        mask_logits_by_level = None
        auxiliary_masks = []
        auxiliary_classes = []
        for layer in self.layers:
            if mask_logits is None:
                attention_mask = torch.zeros(
                    features.shape[0],
                    queries.shape[1],
                    memory.shape[1],
                    dtype=features.dtype,
                    device=features.device,
                )
                if room_token_padding is not None:
                    attention_mask[:, :, -room_token_padding.shape[1] :] = (
                        soft_presence_attention_bias(
                            room_token_padding,
                        )[:, None]
                    )
            else:
                if self.partition_consistent and self.partition_attention_masking:
                    level_masks = torch.cat(
                        [
                            soft_owner_attention_bias(logits)
                            for logits in mask_logits_by_level
                        ],
                        dim=-1,
                    )
                else:
                    level_masks = torch.cat(
                        [
                            soft_pixel_attention_bias(
                                mask_logits,
                                level.shape[-2:],
                            )
                            for level, _memory in levels
                        ],
                        dim=-1,
                    )
                attention_mask = level_masks
                if prototype_tokens is not None and include_prototype_memory:
                    auxiliary = torch.zeros(
                        features.shape[0],
                        queries.shape[1],
                        prototype_tokens.shape[1],
                        dtype=features.dtype,
                        device=features.device,
                    )
                    attention_mask = torch.cat([attention_mask, auxiliary], dim=-1)
                if room_tokens is not None:
                    auxiliary = (
                        soft_presence_attention_bias(
                            room_token_padding,
                        )[:, None].expand(
                            -1,
                            queries.shape[1],
                            -1,
                        )
                        if room_token_padding is not None
                        else torch.zeros(
                            features.shape[0],
                            queries.shape[1],
                            room_tokens.shape[1],
                            dtype=features.dtype,
                            device=features.device,
                        )
                    )
                    attention_mask = torch.cat([attention_mask, auxiliary], dim=-1)
            queries = layer(queries, memory, attention_mask)
            mask_embedding = self.mask_head(queries)
            all_mask_logits = torch.einsum("bqc,bnc->bqn", mask_embedding, memory)
            level_lengths = [item[1].shape[1] for item in levels]
            mask_logits_by_level = []
            offset = 0
            for length in level_lengths:
                mask_logits_by_level.append(
                    all_mask_logits[:, :, offset : offset + length]
                )
                offset += length
            first_level = levels[0][0]
            mask_logits = mask_logits_by_level[0].reshape(
                features.shape[0],
                -1,
                first_level.shape[-2],
                first_level.shape[-1],
            )
            auxiliary_masks.append(mask_logits)
            class_queries = self.class_fusion(queries, program_tokens)
            auxiliary_classes.append(
                self.class_head(self.decoder_norm(class_queries))
            )
        class_queries = self.class_fusion(queries, program_tokens)
        normalized_queries = self.decoder_norm(class_queries)
        class_logits = self.class_head(normalized_queries)
        if self.full_resolution_mask_scale is not None:
            if full_resolution_prototype_logits is None:
                raise ValueError(
                    "full-resolution mask refinement requires prototype logits"
                )
            highres_prototypes = full_resolution_prototype_logits.clone().float()
            highres_tokens = self.prototype_projection(
                highres_prototypes
            ).flatten(2).transpose(1, 2)
            highres_mask = torch.einsum(
                "bqc,bnc->bqn",
                self.mask_head(queries),
                highres_tokens,
            ).reshape(
                features.shape[0],
                queries.shape[1],
                highres_prototypes.shape[-2],
                highres_prototypes.shape[-1],
            )
            mask_logits = F.interpolate(
                mask_logits,
                size=highres_prototypes.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ) + torch.tanh(self.full_resolution_mask_scale) * highres_mask
        mask_class_evidence = None
        if prototype_logits is not None:
            mask_probability = mask_logits.sigmoid()
            evidence_prototypes = prototypes
            if prototypes.shape[-2:] != mask_logits.shape[-2:]:
                evidence_prototypes = F.interpolate(
                    prototypes,
                    size=mask_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            denominator = mask_probability.sum(
                dim=(2, 3),
                keepdim=True,
            ).clamp_min(1.0e-6)
            mask_class_evidence = torch.einsum(
                "bqhw,bchw->bqc",
                mask_probability,
                evidence_prototypes,
            ) / denominator[..., 0]
            class_logits[:, :, :-1] = (
                class_logits[:, :, :-1]
                + torch.tanh(self.fusion_scale).to(class_logits.dtype)
                * mask_class_evidence
            )
        output = {
            "class_logits": class_logits,
            "query_features": normalized_queries,
            "mask_logits": mask_logits,
            "aux_mask_logits": auxiliary_masks,
            "aux_class_logits": auxiliary_classes,
            "fusion_scale": self.fusion_scale,
            "mask_class_evidence": mask_class_evidence,
            "adaptive_query_diagnostics": adaptive_diagnostics,
        }
        output["query_existence_logits"] = query_existence_from_class_logits(
            output["class_logits"]
        )
        if self.partition_consistent:
            output["room_partition_logits"] = mask_logits
            output["room_partition_probabilities"] = mask_logits.softmax(dim=1)
        return output

def query_existence_from_class_logits(
    class_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert foreground/no-object logits into a query presence logit."""

    if class_logits.ndim != 3 or class_logits.shape[-1] < 2:
        raise ValueError("class logits must be [B,Q,C+1]")
    return (
        torch.logsumexp(class_logits[..., :-1], dim=-1)
        - class_logits[..., -1]
    )


def room_mask_targets(
    labels: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
    max_rooms: int = 32,
    min_area: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = labels.shape[-2:]
    classes = []
    masks = []
    valid = []
    for label_map in labels.detach().cpu().numpy():
        instances = sorted(
            extract_room_instances(label_map, taxonomy, min_area=min_area),
            key=lambda item: item.area,
            reverse=True,
        )[:max_rooms]
        batch_classes = torch.zeros(max_rooms, dtype=torch.long)
        batch_masks = torch.zeros(max_rooms, height, width, dtype=torch.float32)
        batch_valid = torch.zeros(max_rooms, dtype=torch.bool)
        for index, instance in enumerate(instances):
            batch_classes[index] = taxonomy.room_ids.index(instance.class_id)
            batch_masks[index] = torch.from_numpy(instance.mask.astype(np.float32))
            batch_valid[index] = True
        classes.append(batch_classes)
        masks.append(batch_masks)
        valid.append(batch_valid)
    return (
        torch.stack(classes).to(labels.device),
        torch.stack(masks).to(labels.device),
        torch.stack(valid).to(labels.device),
    )

@torch.no_grad()
def room_mask_metrics(
    predictions: dict[str, torch.Tensor],
    target_classes: torch.Tensor,
    target_masks: torch.Tensor,
    target_valid: torch.Tensor,
    taxonomy: FloorplanTaxonomy | None = None,
    target_labels: torch.Tensor | None = None,
    footprint: torch.Tensor | None = None,
    *,
    assignment_class_weight: float = 1.0,
    assignment_mask_weight: float = 2.0,
) -> dict[str, float]:
    probability = predictions["mask_logits"].sigmoid()
    ious = []
    correct = 0
    matched = 0
    predicted_classes = []
    for index in range(target_classes.shape[0]):
        count = int(target_valid[index].sum())
        if count == 0:
            continue
        class_probability = predictions["class_logits"][index].softmax(dim=-1)
        class_cost = -class_probability[:, target_classes[index, :count]]
        mask_cost = _pairwise_mask_cost(probability[index], target_masks[index, :count])
        cost = torch.nan_to_num(
            float(assignment_class_weight) * class_cost
            + float(assignment_mask_weight) * mask_cost,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        )
        rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
        predicted_classes.extend(
            int(value) for value in predictions["class_logits"][index, rows].argmax(dim=-1).detach().cpu()
        )
        for row, column in zip(rows, columns):
            predicted = probability[index, row] > 0.5
            expected = target_masks[index, column] > 0.5
            ious.append(float((predicted & expected).sum().float() / (predicted | expected).sum().clamp_min(1).float()))
        correct += int(
            (
                predictions["class_logits"][index, rows].argmax(dim=-1)
                == target_classes[index, columns]
            ).sum()
        )
        matched += len(rows)
    class_counts = np.unique(np.asarray(predicted_classes, dtype=np.int64), return_counts=True)[1]
    class_distribution = class_counts / class_counts.sum() if class_counts.size else np.ones(1)
    result = {
        "mask_iou": float(np.mean(ious)) if ious else 0.0,
        "matched_class_accuracy": correct / max(1, matched),
        "unique_predicted_classes": float(class_counts.size),
        "predicted_class_entropy": float(-(class_distribution * np.log(class_distribution)).sum()),
    }
    if taxonomy is not None:
        rendered = render_room_masks(predictions, taxonomy, footprint)
        rendered_ious = []
        prior_ious = []
        prior_logits = predictions.get("prior_logits")
        prior_labels = None
        if prior_logits is not None and footprint is not None:
            prior_room_ids = torch.tensor(taxonomy.room_ids, device=prior_logits.device)
            prior_labels = prior_room_ids[prior_logits.argmax(dim=1)] * footprint.long()
        if target_labels is not None and footprint is not None:
            dense_target = batch_room_partition_target(target_labels, footprint, taxonomy)
            expected_labels = torch.tensor(taxonomy.room_ids, device=rendered.device)[dense_target.clamp_min(0)]
            expected_valid = dense_target >= 0
        else:
            expected_labels = None
            expected_valid = None
        for batch_index in range(rendered.shape[0]):
            if expected_labels is None:
                expected = target_masks[batch_index].argmax(dim=0) + 1
                valid = target_masks[batch_index].sum(dim=0) > 0
            else:
                expected = expected_labels[batch_index]
                valid = expected_valid[batch_index]
            for class_index in range(len(taxonomy.room_ids)):
                target_mask = (expected == taxonomy.room_ids[class_index]) & valid
                if not target_mask.any():
                    continue
                predicted_mask = rendered[batch_index] == taxonomy.room_ids[class_index]
                rendered_ious.append(
                    float(
                        (target_mask & predicted_mask).sum().float()
                        / (target_mask | predicted_mask).sum().clamp_min(1).float()
                    )
                )
                if prior_labels is not None:
                    prior_mask = prior_labels[batch_index] == taxonomy.room_ids[class_index]
                    prior_ious.append(
                        float(
                            (target_mask & prior_mask).sum().float()
                            / (target_mask | prior_mask).sum().clamp_min(1).float()
                        )
                    )
        result["rendered_room_macro_iou"] = float(np.mean(rendered_ious)) if rendered_ious else 0.0
        if prior_ious:
            result["prior_room_macro_iou"] = float(np.mean(prior_ious))
    return result
