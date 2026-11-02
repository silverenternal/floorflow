"""Typed cross-attention aggregation over independently trained atom evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from floorflow.chip_flow.typed_cross_attention import TypedCrossAttention


@dataclass(frozen=True)
class TypedAtomOutput:
    """Updated atom evidence, reliability and fused candidate utility."""

    atom_updates: torch.Tensor
    atom_logits: torch.Tensor
    atom_gates: torch.Tensor
    aggregate_utility: torch.Tensor
    case_utility: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _broadcast_bool(
    value: torch.Tensor | None,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
    try:
        return torch.broadcast_to(tensor, shape).clone()
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to {shape}") from error


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = logits.shape[-1]
    safe = mask.clone()
    no_atom = ~safe.any(dim=-1)
    if bool(no_atom.any()):
        safe[no_atom, 0] = True
    weights = torch.softmax(logits.masked_fill(~safe, float("-inf")), dim=-1)
    weights = torch.where(
        no_atom[..., None],
        torch.zeros_like(weights),
        weights,
    )
    if weights.shape[-1] != count:
        raise RuntimeError("masked atom softmax changed the atom dimension")
    return weights


class TypedAtomAggregator(nn.Module):
    """Sparse atom/context aggregation with a zero-gated legacy residual.

    ``atom_tokens`` may have leading dimensions ``(batch,)`` or
    ``(batch, candidate)``. Contexts and masks follow the same leading shape.
    Atom-to-context and atom-to-atom attention use the existing sparse
    :class:`TypedCrossAttention`; no dense all-to-all matrix is introduced.
    """

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        atom_count: int = 6,
        context_types: int = 5,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("hidden must be positive and divisible by heads")
        if atom_count <= 0 or context_types <= 0:
            raise ValueError("atom_count and context_types must be positive")
        self.hidden = int(hidden)
        self.atom_count = int(atom_count)
        self.context_types = int(context_types)
        self.context_attention = TypedCrossAttention(hidden, heads)
        self.atom_attention = TypedCrossAttention(hidden, heads)
        self.type_embedding = nn.Parameter(
            torch.zeros(context_types, hidden)
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.utility_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.output_gate = nn.Parameter(torch.zeros(1))
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def _validate_tokens(
        self,
        atom_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if atom_tokens.ndim < 2 or atom_tokens.shape[-2:] != (
            self.atom_count,
            self.hidden,
        ):
            raise ValueError("atom_tokens must end with (atom_count, hidden)")
        leading = atom_tokens.shape[:-2]
        if context_tokens.ndim != len(leading) + 2:
            raise ValueError("context_tokens rank does not match atom_tokens")
        if context_tokens.shape[:-2] != leading or context_tokens.shape[-1] != self.hidden:
            raise ValueError(
                "context_tokens must have atom leading shape and the same hidden width"
            )
        return leading, context_tokens.shape[:-2] + (context_tokens.shape[-2],)

    def forward(
        self,
        atom_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        atom_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        atom_allowed: torch.Tensor | None = None,
        atom_edges: torch.Tensor | None = None,
        context_type_ids: torch.Tensor | None = None,
        legacy_utility: torch.Tensor | None = None,
    ) -> TypedAtomOutput:
        atom_tokens = atom_tokens.float()
        context_tokens = context_tokens.float()
        leading, context_shape = self._validate_tokens(
            atom_tokens,
            context_tokens,
        )
        context_count = context_shape[-1]
        atom_mask_full = _broadcast_bool(
            atom_mask,
            (*leading, self.atom_count),
            atom_tokens.device,
            "atom_mask",
        )
        context_mask_full = _broadcast_bool(
            context_mask,
            (*leading, context_count),
            context_tokens.device,
            "context_mask",
        )
        context_allowed = _broadcast_bool(
            atom_allowed,
            (*leading, self.atom_count, context_count),
            atom_tokens.device,
            "atom_allowed",
        )
        atom_edges_full = _broadcast_bool(
            atom_edges,
            (*leading, self.atom_count, self.atom_count),
            atom_tokens.device,
            "atom_edges",
        )
        context_allowed &= context_mask_full[..., None, :]
        context_allowed &= atom_mask_full[..., :, None]
        atom_edges_full &= atom_mask_full[..., None, :]
        atom_edges_full &= atom_mask_full[..., :, None]
        if context_type_ids is None:
            type_ids = torch.zeros(
                (*leading, context_count),
                dtype=torch.long,
                device=atom_tokens.device,
            )
        else:
            type_ids = torch.as_tensor(
                context_type_ids,
                dtype=torch.long,
                device=atom_tokens.device,
            )
            type_ids = torch.broadcast_to(
                type_ids,
                (*leading, context_count),
            )
        if type_ids.numel() and (
            int(type_ids.min()) < 0
            or int(type_ids.max()) >= self.context_types
        ):
            raise ValueError("context_type_ids are out of range")

        legacy = None
        if legacy_utility is not None:
            legacy = torch.as_tensor(
                legacy_utility,
                dtype=atom_tokens.dtype,
                device=atom_tokens.device,
            )
            try:
                legacy = torch.broadcast_to(legacy, leading)
            except RuntimeError as error:
                raise ValueError(
                    f"legacy_utility cannot broadcast to {leading}"
                ) from error
            legacy = legacy.clone()
        base = (
            legacy
            if legacy is not None
            else torch.zeros(leading, dtype=atom_tokens.dtype, device=atom_tokens.device)
        )

        if not self.enabled:
            atom_updates = atom_tokens
            atom_logits = torch.zeros(
                (*leading, self.atom_count),
                dtype=atom_tokens.dtype,
                device=atom_tokens.device,
            )
            atom_gates = torch.zeros_like(atom_logits)
            diagnostics = {
                "atom_gate_entropy": atom_gates.sum() * 0.0,
                "context_tokens": atom_tokens.sum() * 0.0,
            }
            return TypedAtomOutput(
                atom_updates=atom_updates,
                atom_logits=atom_logits,
                atom_gates=atom_gates,
                aggregate_utility=base,
                case_utility=(
                    base.mean(dim=-1) if base.ndim > 1 else base
                ),
                diagnostics=diagnostics,
            )

        flat_batch = math.prod(leading) if leading else 1
        flat_atoms = atom_tokens.reshape(flat_batch, self.atom_count, self.hidden)
        typed_context = context_tokens + self.type_embedding[type_ids]
        flat_context = typed_context.reshape(flat_batch, context_count, self.hidden)
        flat_context_allowed = context_allowed.reshape(
            flat_batch,
            self.atom_count,
            context_count,
        )
        context_update = self.context_attention(
            flat_atoms,
            flat_context,
            flat_context_allowed,
        )
        context_updates = flat_atoms + context_update
        flat_atom_edges = atom_edges_full.reshape(
            flat_batch,
            self.atom_count,
            self.atom_count,
        )
        atom_update = self.atom_attention(
            context_updates,
            context_updates,
            flat_atom_edges,
        )
        updated = context_updates + atom_update
        atom_updates = updated.reshape(*leading, self.atom_count, self.hidden)
        logits = self.reliability_head(atom_updates).squeeze(-1)
        gates = _masked_softmax(logits, atom_mask_full)
        pooled = (gates[..., None] * atom_updates).sum(dim=-2)
        delta = self.utility_head(pooled).squeeze(-1)
        aggregate = base + self.output_gate * delta
        entropy = -(
            gates.clamp_min(1.0e-8).log() * gates
        ).sum(dim=-1)
        return TypedAtomOutput(
            atom_updates=atom_updates,
            atom_logits=logits,
            atom_gates=gates,
            aggregate_utility=aggregate,
            case_utility=(
                aggregate.mean(dim=-1) if aggregate.ndim > 1 else aggregate
            ),
            diagnostics={
                "atom_gate_entropy": entropy.mean(),
                "context_attention_norm": context_update.float().norm(dim=-1).mean(),
                "atom_attention_norm": atom_update.float().norm(dim=-1).mean(),
            },
        )


__all__ = ["TypedAtomAggregator", "TypedAtomOutput"]
