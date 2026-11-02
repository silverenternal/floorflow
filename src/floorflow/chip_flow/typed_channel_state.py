"""Separated typed channel representations and fused graph outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TypedChannelState:
    """Pre-fusion typed tokens and the sparse routing contract."""

    block_tokens: torch.Tensor
    global_token: torch.Tensor
    structural_tokens: torch.Tensor
    constraint_tokens: torch.Tensor
    port_tokens: torch.Tensor
    structural_mask: torch.Tensor
    constraint_mask: torch.Tensor
    port_mask: torch.Tensor
    port_block_incidence: torch.Tensor
    allowed: torch.Tensor
    group_sizes: tuple[int, ...] | None
    relation_memories: tuple[torch.Tensor, ...] | None
    relation_masks: tuple[torch.Tensor, ...] | None
    diagnostics: dict[str, torch.Tensor]

    def detach(self) -> TypedChannelState:
        """Cut autograd history while retaining all inference fields."""

        return TypedChannelState(
            block_tokens=self.block_tokens.detach(),
            global_token=self.global_token.detach(),
            structural_tokens=self.structural_tokens.detach(),
            constraint_tokens=self.constraint_tokens.detach(),
            port_tokens=self.port_tokens.detach(),
            structural_mask=self.structural_mask.detach(),
            constraint_mask=self.constraint_mask.detach(),
            port_mask=self.port_mask.detach(),
            port_block_incidence=self.port_block_incidence.detach(),
            allowed=self.allowed.detach(),
            group_sizes=self.group_sizes,
            relation_memories=(
                tuple(value.detach() for value in self.relation_memories)
                if self.relation_memories is not None
                else None
            ),
            relation_masks=(
                tuple(value.detach() for value in self.relation_masks)
                if self.relation_masks is not None
                else None
            ),
            diagnostics={
                name: value.detach()
                for name, value in self.diagnostics.items()
            },
        )


@dataclass(frozen=True)
class TypedGraphOutput:
    """Public typed graph encoder output consumed by the hierarchy model."""

    block_tokens: torch.Tensor
    global_token: torch.Tensor
    structural_tokens: torch.Tensor
    constraint_tokens: torch.Tensor
    port_tokens: torch.Tensor
    structural_mask: torch.Tensor
    constraint_mask: torch.Tensor
    port_mask: torch.Tensor
    port_block_incidence: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


__all__ = ["TypedChannelState", "TypedGraphOutput"]
