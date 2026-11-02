"""Optional hierarchy, hyperedge and legalizer branches for chip flow."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.legalizer_response import (
    LegalizerResponseHead,
    apply_legalizer_displacement,
)
from floorflow.chip_flow.netlist_hypergraph import (
    NetlistHyperedgeConditioner,
    NetlistHypergraphContext,
)
from floorflow.chip_flow.structured_heads import (
    MembershipHierarchyCoupler,
    StructuredHeadSuite,
)


class ChipHierarchyConditioners(nn.Module):
    """Keep opt-in structural branches independent and numerically inert at init."""

    def __init__(
        self,
        hidden: int,
        *,
        structured_heads: StructuredHeadSuite | None,
        membership_hierarchy: bool,
        netlist_hyperedge: bool,
        legalizer_response: bool,
    ) -> None:
        super().__init__()
        self.membership = (
            MembershipHierarchyCoupler(hidden)
            if membership_hierarchy
            else None
        )
        if (
            self.membership is not None
            and structured_heads is not None
            and structured_heads.hierarchy_coupling is not None
        ):
            raise ValueError(
                "use either membership_hierarchy or full structured coupling, not both"
            )
        self.hyperedge = (
            NetlistHyperedgeConditioner(hidden, feature_dim=4)
            if netlist_hyperedge
            else None
        )
        self.legalizer_head = (
            LegalizerResponseHead(hidden) if legalizer_response else None
        )

    def residual(
        self,
        latent_residual: torch.Tensor,
        *,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        membership: torch.Tensor | None,
        structural_mask: torch.Tensor,
        global_prediction: torch.Tensor,
        structural_prediction: torch.Tensor,
        structural_tokens: torch.Tensor,
        time: torch.Tensor,
        netlist: NetlistHypergraphContext | None,
        port_state: torch.Tensor,
        port_tokens: torch.Tensor,
        port_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.membership is not None:
            if membership is None:
                raise ValueError("membership hierarchy requires structural membership")
            latent_residual = latent_residual + self.membership(
                block_tokens,
                block_state,
                membership,
                structural_mask,
                global_prediction,
                structural_prediction,
                structural_tokens,
                time,
            )
        if self.hyperedge is not None:
            if netlist is None:
                raise ValueError(
                    "netlist hyperedge conditioning requires a netlist context"
                )
            latent_residual = latent_residual + self.hyperedge(
                block_state,
                block_tokens,
                netlist,
                port_state=port_state,
                port_tokens=port_tokens,
                port_mask=port_mask,
            )
        return latent_residual

    def legalizer_response(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        block_prediction: torch.Tensor,
        time_token: torch.Tensor,
        existing_displacement: torch.Tensor | None,
        existing_rung_logits: torch.Tensor | None,
        apply_action: bool,
        strength: float,
        max_displacement: float,
        center_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        displacement = existing_displacement
        rung_logits = existing_rung_logits
        if self.legalizer_head is not None and displacement is None:
            displacement, rung_logits = self.legalizer_head(
                block_tokens,
                block_state,
                block_prediction,
                time_token,
            )
            if apply_action:
                block_prediction = apply_legalizer_displacement(
                    block_prediction,
                    displacement,
                    strength=strength,
                    max_displacement=max_displacement,
                    center_mask=center_mask,
                )
        return block_prediction, displacement, rung_logits


__all__ = ["ChipHierarchyConditioners"]
