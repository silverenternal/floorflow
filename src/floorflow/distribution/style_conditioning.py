"""Low-rank style conditioning for pixel, element, and velocity distributions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def validate_style_ids(
    style_ids: torch.Tensor | None,
    *,
    batch_size: int,
    num_styles: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate one optional style id per sample and normalize its dtype/device."""

    if style_ids is None:
        return None
    if style_ids.ndim != 1 or style_ids.shape[0] != batch_size:
        raise ValueError("style_ids must be [batch]")
    style_ids = style_ids.to(device=device, dtype=torch.long)
    if int(style_ids.min()) < 0 or int(style_ids.max()) >= int(num_styles):
        raise ValueError("style id is outside the configured style bank")
    return style_ids


class LowRankStyleAdapter(nn.Module):
    """Apply one learned low-rank residual per style without changing base weights."""

    def __init__(self, channels: int, num_styles: int, rank: int = 8) -> None:
        super().__init__()
        if num_styles <= 0 or rank <= 0:
            raise ValueError("num_styles and rank must be positive")
        self.num_styles = int(num_styles)
        self.rank = int(rank)
        self.embedding = nn.Embedding(num_styles, rank)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.normal_(self.up.weight, std=0.01)
        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        features: torch.Tensor,
        style_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        style_ids = validate_style_ids(
            style_ids,
            batch_size=features.shape[0],
            num_styles=self.num_styles,
            device=features.device,
        )
        if style_ids is None:
            return features
        coefficient = self.embedding(style_ids).to(features.dtype)
        if not bool(coefficient.detach().ne(0).any()):
            return features
        residual = self.up(
            self.down(features) * coefficient[:, :, None, None]
        )
        feature_rms = features.detach().float().square().mean().sqrt().clamp_min(
            1.0e-4
        )
        residual_rms = residual.detach().float().square().mean().sqrt().clamp_min(
            1.0e-4
        )
        strength = 0.05 * self.log_strength.exp()
        normalized = residual * (feature_rms / residual_rms).to(residual.dtype)
        return features + strength.to(features.dtype) * normalized


class StylePixelResidual(nn.Module):
    """Learn style-specific foreground-class logit offsets for the pixel head."""

    def __init__(self, num_styles: int, num_foreground_classes: int) -> None:
        super().__init__()
        if num_styles <= 0 or num_foreground_classes <= 0:
            raise ValueError("style and foreground class counts must be positive")
        self.num_styles = int(num_styles)
        self.num_foreground_classes = int(num_foreground_classes)
        self.class_bias = nn.Embedding(
            self.num_styles,
            self.num_foreground_classes,
        )
        nn.init.zeros_(self.class_bias.weight)

    def forward(
        self,
        logits: torch.Tensor,
        style_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        style_ids = validate_style_ids(
            style_ids,
            batch_size=logits.shape[0],
            num_styles=self.num_styles,
            device=logits.device,
        )
        if style_ids is None:
            return logits
        if logits.shape[1] != self.num_foreground_classes + 1:
            raise ValueError("pixel logits must include background plus foreground")
        output = logits.clone()
        output[:, 1:] = output[:, 1:] + self.class_bias(style_ids).to(
            output.dtype
        )[:, :, None, None]
        return output


class StyleElementResidual(nn.Module):
    """Style-specific token, presence, and type residuals for element queries."""

    def __init__(
        self,
        num_styles: int,
        hidden_channels: int,
        group_class_counts: Mapping[str, int],
        queries_per_group: Mapping[str, int],
    ) -> None:
        super().__init__()
        if num_styles <= 0 or hidden_channels <= 0:
            raise ValueError("style and hidden channel counts must be positive")
        self.num_styles = int(num_styles)
        self.hidden_channels = int(hidden_channels)
        self.group_names = tuple(group_class_counts)
        self.query_sizes = {
            name: int(queries_per_group[name]) for name in self.group_names
        }
        self.type_sizes = {
            name: int(group_class_counts[name]) for name in self.group_names
        }
        self.token_bias = nn.Embedding(self.num_styles, self.hidden_channels)
        self.presence_bias = nn.Embedding(
            self.num_styles,
            sum(self.query_sizes.values()),
        )
        self.type_bias = nn.Embedding(
            self.num_styles,
            sum(self.type_sizes.values()),
        )
        for parameter in (
            self.token_bias.weight,
            self.presence_bias.weight,
            self.type_bias.weight,
        ):
            nn.init.zeros_(parameter)
        query_offset = 0
        self.query_slices: dict[str, tuple[int, int]] = {}
        for name in self.group_names:
            size = self.query_sizes[name]
            self.query_slices[name] = (query_offset, query_offset + size)
            query_offset += size
        type_offset = 0
        self.type_slices: dict[str, tuple[int, int]] = {}
        for name in self.group_names:
            size = self.type_sizes[name]
            self.type_slices[name] = (type_offset, type_offset + size)
            type_offset += size

    def _ids(
        self,
        style_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        return validate_style_ids(
            style_ids,
            batch_size=batch_size,
            num_styles=self.num_styles,
            device=device,
        )

    def augment_tokens(
        self,
        tokens: torch.Tensor,
        style_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        style_ids = self._ids(style_ids, tokens.shape[0], tokens.device)
        if style_ids is None:
            return tokens
        bias = self.token_bias(style_ids).to(tokens.dtype)
        return tokens + bias[:, None]

    def adjust_logits(
        self,
        name: str,
        presence_logits: torch.Tensor,
        type_logits: torch.Tensor,
        style_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style_ids = self._ids(
            style_ids,
            presence_logits.shape[0],
            presence_logits.device,
        )
        if style_ids is None:
            return presence_logits, type_logits
        query_start, query_end = self.query_slices[name]
        type_start, type_end = self.type_slices[name]
        presence = presence_logits + self.presence_bias(style_ids)[
            :,
            query_start:query_end,
        ].to(presence_logits.dtype)
        types = type_logits + self.type_bias(style_ids)[
            :,
            type_start:type_end,
        ].to(type_logits.dtype)[:, None]
        return presence, types


def _direction_matrix(module: nn.Module) -> torch.Tensor | None:
    if isinstance(module, LowRankStyleAdapter):
        return module.embedding.weight
    if isinstance(module, StylePixelResidual):
        return module.class_bias.weight
    if isinstance(module, StyleElementResidual):
        return torch.cat(
            [
                module.token_bias.weight,
                module.presence_bias.weight,
                module.type_bias.weight,
            ],
            dim=-1,
        )
    return None


def style_orthogonality_loss(model, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Penalize overlapping style directions within each conditioning path."""

    losses = []
    for module in model.modules():
        directions = _direction_matrix(module)
        if directions is None or directions.shape[0] < 2:
            continue
        directions = directions.float()
        squared_norm = directions.square().sum(
            dim=-1,
            keepdim=True,
        )
        active = squared_norm.detach().flatten() > float(eps) ** 2
        if int(active.sum()) < 2:
            continue
        directions = directions[active]
        squared_norm = directions.square().sum(dim=-1, keepdim=True)
        normalized = directions / squared_norm.clamp_min(
            float(eps) ** 2
        ).sqrt()
        gram = normalized @ normalized.transpose(0, 1)
        eye = torch.eye(
            gram.shape[0],
            device=gram.device,
            dtype=gram.dtype,
        )
        losses.append(((gram - eye) ** 2).mean())
    if not losses:
        reference = next(model.parameters())
        return reference.sum() * 0.0
    return torch.stack(losses).mean()


def style_regularization(
    adapter: LowRankStyleAdapter,
    style_ids: torch.Tensor | None,
    weight: float = 1.0e-4,
) -> torch.Tensor:
    """Small L2 penalty that keeps rarely used style directions bounded."""

    if style_ids is None or weight <= 0:
        return adapter.up.weight.sum() * 0.0
    embedding = adapter.embedding(style_ids.long())
    return float(weight) * embedding.square().mean()


def model_style_contrast_loss(
    model,
    features: torch.Tensor,
    style_ids: torch.Tensor,
    *,
    margin: float = 0.01,
) -> torch.Tensor:
    """Separate assigned and alternate styles on one contour feature map."""

    if model.num_styles <= 1:
        return features.sum() * 0.0
    style_ids = validate_style_ids(
        style_ids,
        batch_size=features.shape[0],
        num_styles=model.num_styles,
        device=features.device,
    )
    if style_ids is None:
        return features.sum() * 0.0
    alternate = (style_ids + 1) % model.num_styles
    adapters = (
        getattr(model.hub.pixel, "style_adapter", None),
        getattr(model.hub.element, "style_adapter", None),
        getattr(model, "velocity_style_adapter", None),
    )
    losses = []
    for adapter in adapters:
        if adapter is None:
            continue
        assigned = adapter(features, style_ids)
        other = adapter(features, alternate)
        difference = (assigned - other).abs().mean()
        scale = features.detach().float().abs().mean().clamp_min(1.0e-4)
        relative = difference / scale
        losses.append(torch.nn.functional.relu(float(margin) - relative))
    return torch.stack(losses).mean() if losses else features.sum() * 0.0


__all__ = [
    "LowRankStyleAdapter",
    "StyleElementResidual",
    "StylePixelResidual",
    "model_style_contrast_loss",
    "style_orthogonality_loss",
    "style_regularization",
    "validate_style_ids",
]
