"""Reusable sparse message and residual blocks for chip token hierarchies."""

from __future__ import annotations

import torch
from torch import nn


def _edge_mask(
    counts: torch.Tensor | None,
    edges: torch.Tensor,
) -> torch.Tensor:
    if counts is None:
        return torch.ones(
            edges.shape[:2],
            device=edges.device,
            dtype=torch.bool,
        )
    rows = torch.arange(edges.shape[1], device=edges.device)[None, :]
    return rows < counts.to(edges.device).long()[:, None]


def masked_token_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average valid tokens with a batched matrix multiply."""

    weight = mask.to(tokens.dtype)
    total = torch.bmm(weight[:, None, :], tokens).squeeze(1)
    return total / weight.sum(dim=1, keepdim=True).clamp_min(1.0)


class ResidualTokenBlock(nn.Module):
    """Pre-norm MLP residual used after sparse token-to-slot routing."""

    def __init__(self, hidden: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * expansion),
            nn.SiLU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.mlp(self.norm(tokens))


class GraphMessageBlock(nn.Module):
    """Vectorized reverse-edge b2b and pin-to-block message passing."""

    def __init__(
        self,
        hidden: int,
        pin_dim: int = 3,
        *,
        edge_gate: bool = False,
        symmetric_edges: bool = False,
    ) -> None:
        super().__init__()
        self.b2b_message = nn.Linear(hidden, hidden)
        self.pin_message = nn.Sequential(
            nn.Linear(pin_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.b2b_gate = None
        self.symmetric_edges = bool(symmetric_edges)
        self.symmetric_gate = (
            nn.Parameter(torch.zeros(1)) if self.symmetric_edges else None
        )
        if edge_gate:
            self.b2b_gate = nn.Sequential(
                nn.Linear(hidden * 2 + 1, hidden // 4),
                nn.SiLU(),
                nn.Linear(hidden // 4, 1),
            )
            nn.init.zeros_(self.b2b_gate[-1].weight)
            nn.init.zeros_(self.b2b_gate[-1].bias)
        self.residual = ResidualTokenBlock(hidden)

    @staticmethod
    def _flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2])

    def _b2b_context(
        self,
        flat_tokens: torch.Tensor,
        b2b: torch.Tensor | None,
        n_b2b: torch.Tensor | None,
        *,
        batch: int,
        count: int,
    ) -> torch.Tensor:
        context = torch.zeros_like(flat_tokens)
        if b2b is None or b2b.shape[1] == 0:
            return context
        source = b2b[..., 0].long()
        target = b2b[..., 1].long()
        weight = b2b[..., 2].to(flat_tokens.dtype).clamp_min(0.0)
        valid = (
            _edge_mask(n_b2b, b2b)
            & (source >= 0)
            & (source < count)
            & (target >= 0)
            & (target < count)
        )
        offsets = (
            torch.arange(batch, device=b2b.device)[:, None] * count
        ).expand(
            batch,
            b2b.shape[1],
        )
        source_flat = (source + offsets).reshape(-1)
        target_flat = (target + offsets).reshape(-1)
        weight_flat = weight.reshape(-1)
        valid_flat = valid.reshape(-1)
        message = torch.zeros_like(flat_tokens)
        sender = flat_tokens.index_select(0, source_flat[valid_flat])
        base_edge_weight = weight_flat[valid_flat]
        edge_weight = base_edge_weight
        if self.b2b_gate is not None:
            receiver = flat_tokens.index_select(0, target_flat[valid_flat])
            gate = 1.0 + torch.tanh(
                self.b2b_gate(
                    torch.cat([sender, receiver, edge_weight[:, None]], dim=-1)
                )
            ).squeeze(-1)
            edge_weight = edge_weight * gate
        message.index_add_(
            0,
            target_flat[valid_flat],
            sender * edge_weight[:, None],
        )
        if self.symmetric_edges:
            reverse_sender = flat_tokens.index_select(
                0,
                target_flat[valid_flat],
            )
            reverse_weight = base_edge_weight
            if self.b2b_gate is not None:
                reverse_gate = 1.0 + torch.tanh(
                    self.b2b_gate(
                        torch.cat(
                            [
                                reverse_sender,
                                sender,
                                base_edge_weight[:, None],
                            ],
                            dim=-1,
                        )
                    )
                ).squeeze(-1)
                reverse_weight = reverse_weight * reverse_gate
            reverse_message = torch.zeros_like(message)
            reverse_message.index_add_(
                0,
                source_flat[valid_flat],
                reverse_sender * reverse_weight[:, None],
            )
        degree = torch.zeros(
            flat_tokens.shape[0],
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        )
        degree.index_add_(0, target_flat[valid_flat], weight_flat[valid_flat])
        degree.index_add_(0, source_flat[valid_flat], weight_flat[valid_flat])
        if self.symmetric_edges:
            message = message + self.symmetric_gate * reverse_message
        normalized = message / degree[:, None].clamp_min(1.0)
        return context + self.b2b_message(normalized).masked_fill(
            degree[:, None] <= 0.0,
            0.0,
        )

    def _pin_context(
        self,
        flat_tokens: torch.Tensor,
        p2b: torch.Tensor | None,
        pins: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
        n_pins: torch.Tensor | None,
        *,
        batch: int,
        count: int,
    ) -> torch.Tensor:
        context = torch.zeros_like(flat_tokens)
        if p2b is None or pins is None or p2b.shape[1] == 0 or pins.shape[1] == 0:
            return context
        pin_index = p2b[..., 0].long()
        block_index = p2b[..., 1].long()
        weight = p2b[..., 2].to(flat_tokens.dtype).clamp_min(0.0)
        pin_count = (
            n_pins.to(pins.device).long()
            if n_pins is not None
            else torch.full(
                (batch,),
                pins.shape[1],
                device=pins.device,
                dtype=torch.long,
            )
        )
        valid = (
            _edge_mask(n_p2b, p2b)
            & (pin_index >= 0)
            & (pin_index < pin_count[:, None])
            & (block_index >= 0)
            & (block_index < count)
            & (weight > 0.0)
        )
        batch_offset = torch.arange(
            batch,
            device=p2b.device,
        )[:, None]
        pin_flat = (pin_index + batch_offset * pins.shape[1]).reshape(-1)
        block_flat = (block_index + batch_offset * count).reshape(-1)
        valid_flat = valid.reshape(-1)
        flat_pins = pins.to(flat_tokens.dtype).reshape(-1, pins.shape[-1])
        scale = flat_pins.reshape(batch, pins.shape[1], -1).abs().amax(
            dim=(1, 2)
        ).clamp_min(1.0)
        weight_flat = weight.reshape(-1)
        batch_flat = batch_offset.expand(batch, p2b.shape[1]).reshape(-1)
        edge_scale = scale[batch_flat[valid_flat]]
        message = weight_flat[valid_flat, None] * self.pin_message(
            torch.cat(
                [
                    flat_pins.index_select(0, pin_flat[valid_flat])
                    / edge_scale[:, None],
                    weight_flat[valid_flat, None],
                ],
                dim=-1,
            )
        )
        pin_context = torch.zeros_like(flat_tokens)
        pin_context.index_add_(0, block_flat[valid_flat], message)
        degree = torch.zeros(
            flat_tokens.shape[0],
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        )
        degree.index_add_(0, block_flat[valid_flat], weight_flat[valid_flat])
        return context + pin_context / degree[:, None].clamp_min(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        b2b: torch.Tensor | None,
        p2b: torch.Tensor | None,
        pins: torch.Tensor | None,
        n_b2b: torch.Tensor | None,
        n_p2b: torch.Tensor | None,
        n_pins: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, count, _ = tokens.shape
        flat = self._flatten_tokens(tokens)
        context = self._b2b_context(
            flat,
            b2b,
            n_b2b,
            batch=batch,
            count=count,
        )
        context = context + self._pin_context(
            flat,
            p2b,
            pins,
            n_p2b,
            n_pins,
            batch=batch,
            count=count,
        )
        mixed = flat + context
        return self.residual(mixed.reshape(batch, count, -1))


class StateEdgeRelay(nn.Module):
    """Sparse current-state geometry correction over active chip edges."""

    def __init__(self, state_dim: int = 3, hidden: int = 48) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden = int(hidden)
        b2b_features = state_dim * 4 + 3
        p2b_features = state_dim + 5
        self.b2b_encoder = nn.Sequential(
            nn.LayerNorm(b2b_features),
            nn.Linear(b2b_features, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.p2b_encoder = nn.Sequential(
            nn.LayerNorm(p2b_features),
            nn.Linear(p2b_features, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )
        self.output = nn.Linear(hidden, state_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def _b2b(
        self,
        flat_state: torch.Tensor,
        source_rows: torch.Tensor,
        target_rows: torch.Tensor,
        edge_weight: torch.Tensor,
        flat_area: torch.Tensor,
    ) -> torch.Tensor:
        context = flat_state.new_zeros((flat_state.shape[0], self.hidden))
        if source_rows.numel() == 0:
            return context
        source_state = flat_state.index_select(0, source_rows)
        target_state = flat_state.index_select(0, target_rows)
        source_area = flat_area.index_select(0, source_rows)
        target_area = flat_area.index_select(0, target_rows)
        edge_weight = edge_weight[:, None]

        def edge_features(
            left: torch.Tensor,
            right: torch.Tensor,
            left_area: torch.Tensor,
            right_area: torch.Tensor,
        ) -> torch.Tensor:
            delta = right - left
            return torch.cat(
                [
                    left,
                    right,
                    delta,
                    delta.abs(),
                    left_area,
                    right_area,
                    edge_weight,
                ],
                dim=-1,
            )

        forward = self.b2b_encoder(
            edge_features(source_state, target_state, source_area, target_area)
        )
        reverse = self.b2b_encoder(
            edge_features(target_state, source_state, target_area, source_area)
        )
        messages = flat_state.new_zeros(
            (flat_state.shape[0], self.hidden),
            dtype=forward.dtype,
        )
        degree = torch.zeros(
            flat_state.shape[0],
            device=flat_state.device,
            dtype=forward.dtype,
        )
        messages.index_add_(0, target_rows, forward)
        messages.index_add_(0, source_rows, reverse)
        degree_weight = edge_weight.squeeze(-1).to(forward.dtype)
        degree.index_add_(0, target_rows, degree_weight)
        degree.index_add_(0, source_rows, degree_weight)
        return context + messages / degree[:, None].clamp_min(1.0)

    def _p2b(
        self,
        flat_state: torch.Tensor,
        pins: torch.Tensor,
        pin_rows: torch.Tensor,
        block_rows: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        context = flat_state.new_zeros((flat_state.shape[0], self.hidden))
        if pin_rows.numel() == 0:
            return context
        pin_xy = pins.to(flat_state.dtype).reshape(-1, pins.shape[-1])
        pin_state = pin_xy.index_select(0, pin_rows)
        block_state = flat_state.index_select(0, block_rows)
        edge_weight = edge_weight[:, None]
        messages = self.p2b_encoder(
            torch.cat(
                [
                    pin_state,
                    block_state,
                    block_state[..., :2] - pin_state,
                    edge_weight,
                ],
                dim=-1,
            )
        )
        output = flat_state.new_zeros(
            (flat_state.shape[0], self.hidden),
            dtype=messages.dtype,
        )
        degree = torch.zeros(
            flat_state.shape[0],
            device=flat_state.device,
            dtype=messages.dtype,
        )
        output.index_add_(0, block_rows, messages)
        degree.index_add_(
            0,
            block_rows,
            edge_weight.squeeze(-1).to(messages.dtype),
        )
        return context + output / degree[:, None].clamp_min(1.0)

    def forward(
        self,
        state: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch, count, _ = state.shape
        flat_state = state.reshape(batch * count, self.state_dim)
        context = self._b2b(
            flat_state,
            cache["b2b_source"],
            cache["b2b_target"],
            cache["b2b_weight"],
            cache.get("normalized_areas", cache["areas"]),
        )
        context = context + self._p2b(
            flat_state,
            cache.get("normalized_pins", cache["pins"]),
            cache["p2b_pin"],
            cache["p2b_block"],
            cache["p2b_weight"],
        )
        correction = self.gate * self.output(context)
        correction = correction.reshape(batch, count, self.state_dim)
        return correction.to(dtype=state.dtype)


def prepare_state_edge_cache(
    *,
    areas: torch.Tensor,
    b2b: torch.Tensor | None,
    p2b: torch.Tensor | None,
    pins: torch.Tensor | None,
    n_b2b: torch.Tensor | None,
    n_p2b: torch.Tensor | None,
    n_pins: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Precompute active sparse edge rows once per trajectory context."""

    batch, count = areas.shape
    total_area = areas.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    area_fraction = areas / total_area
    flow_scale = total_area.sqrt()
    empty_long = areas.new_empty(0, dtype=torch.long)
    empty_float = areas.new_empty(0)
    b2b_source = empty_long
    b2b_target = empty_long
    b2b_weight = empty_float
    if b2b is not None and b2b.shape[1] > 0:
        source = b2b[..., 0].long()
        target = b2b[..., 1].long()
        weight = b2b[..., 2].to(areas.dtype).clamp_min(0.0)
        valid = (
            _edge_mask(n_b2b, b2b)
            & (source >= 0)
            & (source < count)
            & (target >= 0)
            & (target < count)
            & (weight > 0.0)
        )
        offsets = (
            torch.arange(batch, device=b2b.device)[:, None] * count
        ).expand_as(source)
        valid_flat = valid.reshape(-1)
        b2b_source = (source + offsets).reshape(-1)[valid_flat]
        b2b_target = (target + offsets).reshape(-1)[valid_flat]
        b2b_weight = weight.reshape(-1)[valid_flat]
    empty_pin = empty_long
    p2b_pin = empty_pin
    p2b_block = empty_long
    p2b_weight = empty_float
    flat_pins = (
        pins.to(areas.dtype).reshape(-1, pins.shape[-1])
        if pins is not None
        else areas.new_empty(0, 2)
    )
    normalized_pins = (
        pins.to(areas.dtype).reshape(batch, pins.shape[1], 2)
        / flow_scale[:, None]
    ).reshape(-1, 2) if pins is not None else flat_pins
    if pins is not None and p2b is not None and pins.shape[1] > 0 and p2b.shape[1] > 0:
        pin_index = p2b[..., 0].long()
        block_index = p2b[..., 1].long()
        weight = p2b[..., 2].to(areas.dtype).clamp_min(0.0)
        pin_count = (
            n_pins.to(pins.device).long()
            if n_pins is not None
            else torch.full(
                (batch,),
                pins.shape[1],
                device=pins.device,
                dtype=torch.long,
            )
        )
        valid = (
            _edge_mask(n_p2b, p2b)
            & (pin_index >= 0)
            & (pin_index < pin_count[:, None])
            & (block_index >= 0)
            & (block_index < count)
            & (weight > 0.0)
        )
        offsets = torch.arange(batch, device=p2b.device)[:, None]
        valid_flat = valid.reshape(-1)
        p2b_pin = (
            pin_index + offsets * pins.shape[1]
        ).reshape(-1)[valid_flat]
        p2b_block = (
            block_index + offsets * count
        ).reshape(-1)[valid_flat]
        p2b_weight = weight.reshape(-1)[valid_flat]
    return {
        "areas": areas.reshape(-1, 1),
        "normalized_areas": area_fraction.reshape(-1, 1),
        "pins": flat_pins,
        "normalized_pins": normalized_pins,
        "pin_lengths": (
            n_pins.to(areas.device).long()
            if n_pins is not None
            else torch.full(
                (batch,),
                pins.shape[1] if pins is not None else 0,
                device=areas.device,
                dtype=torch.long,
            )
        ),
        "b2b_source": b2b_source,
        "b2b_target": b2b_target,
        "b2b_weight": b2b_weight,
        "p2b_pin": p2b_pin,
        "p2b_block": p2b_block,
        "p2b_weight": p2b_weight,
    }


__all__ = [
    "GraphMessageBlock",
    "ResidualTokenBlock",
    "StateEdgeRelay",
    "masked_token_mean",
    "prepare_state_edge_cache",
]
