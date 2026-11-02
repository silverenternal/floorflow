"""Optional autoregressive head over the canonical token grammar."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.objectives.token_grammar import TOKEN_STAGES
from floorflow.representation.tensor_checks import assert_true_async


@dataclass
class FloorplanTokenKVCache:
    self_keys: list[torch.Tensor] = field(default_factory=list)
    self_values: list[torch.Tensor] = field(default_factory=list)
    cross_keys: list[torch.Tensor] = field(default_factory=list)
    cross_values: list[torch.Tensor] = field(default_factory=list)
    cross_valid: torch.Tensor | None = None
    next_position: int = 0

    def clone(self) -> FloorplanTokenKVCache:
        return FloorplanTokenKVCache(
            [value.detach().clone() for value in self.self_keys],
            [value.detach().clone() for value in self.self_values],
            [value.detach().clone() for value in self.cross_keys],
            [value.detach().clone() for value in self.cross_values],
            (
                self.cross_valid.detach().clone()
                if self.cross_valid is not None
                else None
            ),
            self.next_position,
        )

    @classmethod
    def stack(
        cls,
        caches: list[FloorplanTokenKVCache],
    ) -> FloorplanTokenKVCache:
        if not caches:
            raise ValueError("at least one cache is required")
        positions = {cache.next_position for cache in caches}
        if len(positions) != 1:
            raise ValueError("beam caches must share a decode position")
        layers = len(caches[0].self_keys)
        if any(len(cache.self_keys) != layers for cache in caches):
            raise ValueError("beam caches must share layers")
        return cls(
            self_keys=[
                torch.cat(
                    [cache.self_keys[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layers)
            ],
            self_values=[
                torch.cat(
                    [cache.self_values[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layers)
            ],
            cross_keys=[
                torch.cat(
                    [cache.cross_keys[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layers)
            ],
            cross_values=[
                torch.cat(
                    [cache.cross_values[layer] for cache in caches],
                    dim=0,
                )
                for layer in range(layers)
            ],
            cross_valid=(
                torch.cat(
                    [cache.cross_valid for cache in caches],
                    dim=0,
                )
                if caches[0].cross_valid is not None
                else None
            ),
            next_position=caches[0].next_position,
        )

    def index_select(self, indices: torch.Tensor) -> FloorplanTokenKVCache:
        return FloorplanTokenKVCache(
            [value.index_select(0, indices) for value in self.self_keys],
            [value.index_select(0, indices) for value in self.self_values],
            [value.index_select(0, indices) for value in self.cross_keys],
            [value.index_select(0, indices) for value in self.cross_values],
            (
                self.cross_valid.index_select(0, indices)
                if self.cross_valid is not None
                else None
            ),
            self.next_position,
        )


def _project_query(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
) -> torch.Tensor:
    batch, _, hidden = query.shape
    heads = module.num_heads
    head_dim = hidden // heads
    query, _key, _value = F.linear(
        query,
        module.in_proj_weight,
        module.in_proj_bias,
    ).chunk(3, dim=-1)
    query = query.view(
        batch,
        1,
        heads,
        head_dim,
    ).transpose(1, 2)
    return query


def _project_key_value(
    module: nn.MultiheadAttention,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, hidden = values.shape
    heads = module.num_heads
    head_dim = hidden // heads
    _query, key, value = F.linear(
        values,
        module.in_proj_weight,
        module.in_proj_bias,
    ).chunk(3, dim=-1)
    key = key.view(batch, -1, heads, head_dim).transpose(1, 2)
    value = value.view(batch, -1, heads, head_dim).transpose(1, 2)
    return key, value


def _finish_attention(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is not None:
        attention_bias = torch.zeros(
            *valid.shape,
            device=valid.device,
            dtype=query.dtype,
        )
        attention_bias = attention_bias.masked_fill(~valid, -torch.inf)
        attention_bias = attention_bias[:, None, None].expand(
            -1,
            module.num_heads,
            1,
            -1,
        )
    else:
        attention_bias = None
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
    )
    output = output.transpose(1, 2).reshape(query.shape[0], 1, -1)
    return module.out_proj(output)


class FloorplanTokenHead(nn.Module):
    def __init__(
        self,
        context_dim: int,
        vocab_size: int,
        max_length: int,
        *,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim % heads:
            heads = 1
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.bos_id = self.vocab_size
        self.pad_id = self.vocab_size + 1
        self.context_projection = nn.Linear(int(context_dim), hidden_dim)
        self.token = nn.Embedding(self.vocab_size + 2, hidden_dim)
        self.position = nn.Embedding(self.max_length, hidden_dim)
        self.stage = nn.Embedding(len(TOKEN_STAGES), hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    hidden_dim,
                    heads,
                    dim_feedforward=hidden_dim * 2,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.vocab_size)

    def forward_with_hidden(
        self,
        context: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim == 2:
            context = context[:, None]
        if context.ndim != 3 or token_ids.ndim != 2:
            raise ValueError("context and token ids must be [B,M,D]/[B,L]")
        if context.shape[0] != token_ids.shape[0]:
            raise ValueError("context and token batches must match")
        if token_ids.shape[1] == 0:
            raise ValueError("token sequence must not be empty")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("token length exceeds max_length")
        if stage_ids is None:
            stage_ids = torch.zeros_like(token_ids)
        if stage_ids.shape != token_ids.shape:
            raise ValueError("stage ids must match token ids")
        if stage_ids.numel() and (
            stage_ids.min() < 0 or stage_ids.max() >= len(TOKEN_STAGES)
        ):
            raise ValueError("stage ids must index TOKEN_STAGES")
        if token_mask is None:
            token_mask = (token_ids >= 0) & (token_ids < self.vocab_size)
        if token_mask.shape != token_ids.shape or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean and match token ids")
        valid_ids = token_ids[token_mask]
        if valid_ids.numel() and (
            valid_ids.min() < 0 or valid_ids.max() >= self.vocab_size
        ):
            raise ValueError("valid token ids must be in the output vocabulary")
        positions = torch.arange(
            token_ids.shape[1],
            device=token_ids.device,
        )
        decoder_input = torch.full_like(token_ids, self.pad_id)
        decoder_input[:, 0] = self.bos_id
        if token_ids.shape[1] > 1:
            previous = token_ids[:, :-1].clamp(0, self.vocab_size - 1)
            decoder_input[:, 1:] = previous.masked_fill(
                ~token_mask[:, :-1],
                self.pad_id,
            )
        input_mask = torch.ones_like(token_mask)
        if token_mask.shape[1] > 1:
            input_mask[:, 1:] = token_mask[:, :-1]
        causal_mask = torch.triu(
            torch.ones(
                token_ids.shape[1],
                token_ids.shape[1],
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = (
            self.token(decoder_input)
            + self.position(positions)[None]
            + self.stage(stage_ids.long())
        )
        memory = self.context_projection(context)
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2],
                dtype=torch.bool,
                device=context.device,
            )
        if context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be boolean [B,M]")
        assert_true_async(
            context_mask.any(dim=1).all(),
            "every sample needs at least one valid context token",
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=~input_mask,
                memory_key_padding_mask=~context_mask,
            )
        normalized = self.norm(hidden)
        return normalized, self.output(normalized)

    def forward(
        self,
        context: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_hidden(
            context,
            token_ids,
            token_mask,
            stage_ids,
            context_mask,
        )[1]

    def incremental_step(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        input_token_ids: torch.Tensor,
        stage_ids: torch.Tensor,
        cache: FloorplanTokenKVCache | None = None,
    ) -> tuple[torch.Tensor, FloorplanTokenKVCache]:
        """Decode one position while reusing past self/cross attention K/V."""

        if context.ndim == 2:
            context = context[:, None]
        if input_token_ids.ndim == 1:
            input_token_ids = input_token_ids[:, None]
        if stage_ids.ndim == 1:
            stage_ids = stage_ids[:, None]
        cache = cache or FloorplanTokenKVCache()
        position = cache.next_position
        if position >= self.max_length:
            raise ValueError("incremental decode exceeds max_length")
        if not cache.cross_keys:
            memory = self.context_projection(context)
            cache.cross_valid = context_mask
            for layer in self.layers:
                key, value = _project_key_value(layer.multihead_attn, memory)
                cache.cross_keys.append(key)
                cache.cross_values.append(value)
        hidden = (
            self.token(input_token_ids)
            + self.position.weight[position][None, None]
            + self.stage(stage_ids.long())
        )
        for index, layer in enumerate(self.layers):
            residual = hidden
            normalized = layer.norm1(hidden)
            key, value = _project_key_value(layer.self_attn, normalized)
            if len(cache.self_keys) <= index:
                cache.self_keys.append(key)
                cache.self_values.append(value)
            else:
                cache.self_keys[index] = torch.cat(
                    [cache.self_keys[index], key],
                    dim=2,
                )
                cache.self_values[index] = torch.cat(
                    [cache.self_values[index], value],
                    dim=2,
                )
            query = _project_query(layer.self_attn, normalized)
            hidden = residual + _finish_attention(
                layer.self_attn,
                query,
                cache.self_keys[index],
                cache.self_values[index],
            )
            residual = hidden
            normalized = layer.norm2(hidden)
            query = _project_query(layer.multihead_attn, normalized)
            hidden = residual + _finish_attention(
                layer.multihead_attn,
                query,
                cache.cross_keys[index],
                cache.cross_values[index],
                valid=cache.cross_valid,
            )
            residual = hidden
            normalized = layer.norm3(hidden)
            feed_forward = layer.linear2(
                layer.activation(layer.linear1(normalized))
            )
            hidden = residual + feed_forward
        cache.next_position += 1
        return self.output(self.norm(hidden))[:, 0], cache


__all__ = ["FloorplanTokenHead"]
