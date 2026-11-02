"""Opt-in structured prediction heads for the chip hierarchy flow."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.chip_flow.legalizer_response import LegalizerResponseHead
from floorflow.chip_flow.sparse_layers import masked_token_mean
from floorflow.chip_flow.typed_cross_attention import TypedCrossAttention


def _zero_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MembershipHierarchyCoupler(nn.Module):
    """Route predicted L0/L1 state to L2 through block membership."""

    def __init__(
        self,
        hidden: int,
        *,
        structural_slots: int = 9,
        bottleneck: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or structural_slots <= 0:
            raise ValueError("hidden and structural_slots must be positive")
        width = int(bottleneck or max(16, hidden // 8))
        self.hidden = int(hidden)
        self.structural_slots = int(structural_slots)
        self.width = width
        self.global_encoder = nn.Linear(4, width)
        self.structure_state_encoder = nn.Linear(3, width)
        self.structure_token_encoder = nn.Linear(hidden, width)
        self.block_token_encoder = nn.Linear(hidden, width)
        self.block_state_encoder = nn.Linear(3, width)
        self.time_encoder = nn.Linear(1, width)
        self.global_reduce = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            _zero_linear(width, 3),
        )
        self.structure_reduce = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            _zero_linear(width, 3),
        )
        # The residual reductions are zero-initialized. Unit gates preserve an
        # exact identity while allowing gradients into both residual heads.
        self.global_gate = nn.Parameter(torch.ones(1))
        self.structure_gate = nn.Parameter(torch.ones(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        membership: torch.Tensor,
        structural_mask: torch.Tensor,
        global_prediction: torch.Tensor,
        structural_prediction: torch.Tensor,
        structural_tokens: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if block_state.shape != (batch, count, 3):
            raise ValueError("block state must have shape (batch, n, 3)")
        if membership.shape != (batch, count, self.structural_slots):
            raise ValueError("membership has the wrong structural-slot count")
        if structural_mask.shape != (batch, self.structural_slots):
            raise ValueError("structural mask must have shape (batch, slots)")
        if structural_prediction.shape != (batch, self.structural_slots, 3):
            raise ValueError("structural prediction has the wrong shape")
        if structural_tokens.shape != (
            batch,
            self.structural_slots,
            hidden,
        ):
            raise ValueError("structural token shape mismatch")
        if global_prediction.shape != (batch, 4):
            raise ValueError("global prediction must have shape (batch, 4)")

        valid = structural_mask[:, None, :].to(membership.dtype)
        weights = membership.to(block_tokens.dtype) * valid
        membership_mass = weights.sum(dim=-1)
        active = membership_mass.gt(0.0)
        denominator = membership_mass[:, :, None].clamp_min(1.0)
        pooled_state = torch.bmm(weights, structural_prediction.float())
        pooled_tokens = torch.bmm(weights, structural_tokens.float())
        pooled_state = pooled_state / denominator
        pooled_tokens = pooled_tokens / denominator

        block_codes = self.block_token_encoder(block_tokens.float())
        state_codes = self.block_state_encoder(block_state.float())
        time_codes = self.time_encoder(time.float()[:, None])
        time_codes = time_codes[:, None].expand(-1, count, -1)
        global_codes = self.global_encoder(global_prediction.float())
        global_codes = global_codes[:, None].expand(-1, count, -1)
        structure_codes = self.structure_state_encoder(pooled_state.float())
        token_codes = self.structure_token_encoder(pooled_tokens.float())

        global_residual = self.global_reduce(
            torch.cat(
                [global_codes, block_codes, state_codes, time_codes],
                dim=-1,
            )
        )
        structural_residual = self.structure_reduce(
            torch.cat(
                [structure_codes, token_codes, state_codes, time_codes],
                dim=-1,
            )
        )
        residual = self.global_gate * global_residual
        residual = residual + (
            self.structure_gate
            * structural_residual
            * active[..., None].to(structural_residual.dtype)
        )
        return residual.to(block_tokens.dtype)


class FactorizedBlockHead(nn.Module):
    """Separate center and rho/aspect corrections over a shared backbone."""

    def __init__(
        self,
        hidden: int,
        *,
        bottleneck: int | None = None,
        shared_trunk: bool = False,
        rho_trust_radius: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if rho_trust_radius < 0.0:
            raise ValueError("rho trust radius must be non-negative")
        width = int(bottleneck or max(16, hidden // 8))
        self.hidden = int(hidden)
        self.width = width
        self.token_encoder = nn.Linear(hidden, width)
        self.state_encoder = nn.Linear(3, width)
        self.base_center_encoder = nn.Linear(2, width)
        self.base_rho_encoder = nn.Linear(1, width)
        self.time_encoder = nn.Linear(hidden, width)
        self.shared_trunk = bool(shared_trunk)
        self.rho_trust_radius = float(rho_trust_radius)
        self.shared = (
            nn.Sequential(nn.Linear(5 * width, 4 * width), nn.SiLU())
            if self.shared_trunk
            else None
        )
        head_input = 4 * width
        self.center_head = nn.Sequential(
            nn.Linear(head_input, width),
            nn.SiLU(),
            _zero_linear(width, 2),
        )
        self.rho_head = nn.Sequential(
            nn.Linear(head_input, width),
            nn.SiLU(),
            _zero_linear(width, 1),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        time_token: torch.Tensor,
        constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, count, hidden = block_tokens.shape
        if hidden != self.hidden:
            raise ValueError("block token hidden size mismatch")
        if base_prediction.shape != (batch, count, 3):
            raise ValueError("base prediction must have shape (batch, n, 3)")
        if block_state.shape != base_prediction.shape:
            raise ValueError("block state must match base prediction")
        token_code = self.token_encoder(block_tokens.float())
        state_code = self.state_encoder(block_state.float())
        time_code = self.time_encoder(time_token.float())
        time_code = time_code[:, None].expand(-1, count, -1)
        center_code = self.base_center_encoder(base_prediction[..., :2].float())
        rho_code = self.base_rho_encoder(base_prediction[..., 2:3].float())
        if self.shared is not None:
            features = self.shared(
                torch.cat(
                    [token_code, state_code, center_code, rho_code, time_code],
                    dim=-1,
                )
            )
            center = self.center_head(features)
            rho = self.rho_head(features)
        else:
            center = self.center_head(
                torch.cat(
                    [token_code, state_code, center_code, time_code],
                    dim=-1,
                )
            )
            rho = self.rho_head(
                torch.cat(
                    [token_code, state_code, rho_code, time_code],
                    dim=-1,
                )
            )
        if self.rho_trust_radius > 0.0:
            rho = rho.clamp(-self.rho_trust_radius, self.rho_trust_radius)
        residual = torch.cat([center, rho], dim=-1)
        if constraints is not None:
            if constraints.shape != (batch, count, 5):
                raise ValueError("constraints must have shape [B,N,5]")
            preplaced = constraints[..., 1].ne(0)
            fixed = constraints[..., 0].ne(0)
            residual = torch.cat(
                [
                    residual[..., :2]
                    * (~preplaced)[..., None].to(residual.dtype),
                    residual[..., 2:]
                    * (~(fixed | preplaced))[..., None].to(residual.dtype),
                ],
                dim=-1,
            )
        return residual.to(base_prediction.dtype)


class DualSemanticPortHead(nn.Module):
    """Separate connected pin offsets from unconnected pin alignment residuals."""

    def __init__(self, hidden: int, *, bottleneck: int | None = None) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        width = int(bottleneck or max(16, hidden // 8))
        self.hidden = int(hidden)
        self.width = width
        self.token_encoder = nn.Linear(hidden, width)
        self.state_encoder = nn.Linear(2, width)
        self.base_encoder = nn.Linear(2, width)
        self.time_encoder = nn.Linear(hidden, width)
        self.connected_head = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            _zero_linear(width, 2),
        )
        self.unconnected_head = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            _zero_linear(width, 2),
        )

    def forward(
        self,
        base_prediction: torch.Tensor,
        port_state: torch.Tensor,
        port_tokens: torch.Tensor,
        connected_mask: torch.Tensor,
        port_mask: torch.Tensor,
        time_token: torch.Tensor,
    ) -> torch.Tensor:
        batch, port_count, hidden = port_tokens.shape
        if hidden != self.hidden:
            raise ValueError("port token hidden size mismatch")
        if base_prediction.shape != (batch, port_count, 2):
            raise ValueError("base port prediction has the wrong shape")
        if port_state.shape != base_prediction.shape:
            raise ValueError("port state must match base port prediction")
        if connected_mask.shape != (batch, port_count):
            raise ValueError("connected mask has the wrong shape")
        if port_mask.shape != (batch, port_count):
            raise ValueError("port mask has the wrong shape")
        token_code = self.token_encoder(port_tokens.float())
        state_code = self.state_encoder(port_state.float())
        base_code = self.base_encoder(base_prediction.float())
        time_code = self.time_encoder(time_token.float())
        time_code = time_code[:, None].expand(-1, port_count, -1)
        features = torch.cat([token_code, state_code, base_code, time_code], dim=-1)
        connected = self.connected_head(features)
        unconnected = self.unconnected_head(features)
        residual = torch.where(connected_mask[..., None], connected, unconnected)
        return (
            base_prediction
            + residual.to(base_prediction.dtype) * port_mask[..., None]
        )


class AtomResidualAggregator(nn.Module):
    """Typed attention across independently trained trajectory residuals."""

    def __init__(
        self,
        hidden: int,
        *,
        atom_count: int = 7,
        heads: int | None = None,
        context_types: int = 5,
    ) -> None:
        super().__init__()
        if hidden <= 0 or atom_count <= 0 or context_types <= 0:
            raise ValueError("invalid atom aggregator dimensions")
        attention_heads = int(heads or max(1, min(4, hidden // 32)))
        while hidden % attention_heads:
            attention_heads -= 1
        self.hidden = int(hidden)
        self.atom_count = int(atom_count)
        self.context_types = int(context_types)
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.block_encoder = nn.Linear(hidden, hidden)
        self.global_encoder = nn.Linear(4, hidden)
        self.attention = TypedCrossAttention(hidden, attention_heads)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.normal_(self.score[-1].weight, std=0.02)
        nn.init.zeros_(self.score[-1].bias)
        self.reliability = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        # ``2 * sigmoid(0) == 1`` preserves the exact residual sum at
        # initialization while allowing fusion to suppress a bad atom.
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)
        self.type_embedding = nn.Parameter(
            torch.randn(atom_count, hidden) * 0.02
        )
        allowed = torch.tensor(
            [
                [1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
        self.register_buffer("allowed", allowed[None])
        atom_edges = torch.ones(atom_count, atom_count, dtype=torch.bool)
        self.register_buffer("atom_edges", atom_edges[None])
        self.mix_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_residuals: torch.Tensor,
        block_tokens: torch.Tensor,
        global_prediction: torch.Tensor,
        structural_tokens: torch.Tensor,
        structural_mask: torch.Tensor,
        port_tokens: torch.Tensor,
        port_mask: torch.Tensor,
        time: torch.Tensor,
        branch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if branch_residuals.shape[1:] != (
            self.atom_count,
            block_tokens.shape[1],
            3,
        ):
            raise ValueError("branch residuals must be [B,atoms,N,3]")
        if branch_mask is None:
            branch_mask = branch_residuals.new_ones(
                branch_residuals.shape[:2],
                dtype=torch.bool,
            )
        else:
            branch_mask = torch.as_tensor(
                branch_mask,
                dtype=torch.bool,
                device=branch_residuals.device,
            ).expand(branch_residuals.shape[0], self.atom_count)
        branch_residuals = branch_residuals * branch_mask[
            :, :, None, None
        ].to(branch_residuals.dtype)
        atom_tokens = (
            self.residual_encoder(branch_residuals.float())
            + self.block_encoder(block_tokens.float())[:, None]
            + self.type_embedding[None, :, None]
        )
        batch, atoms, count, _ = atom_tokens.shape
        structural_mean = masked_token_mean(
            structural_tokens,
            structural_mask,
        )
        port_mean = masked_token_mean(port_tokens, port_mask)
        time_token = time.float()[:, None].expand(-1, self.hidden)
        context = torch.stack(
            [
                block_tokens.float(),
                self.global_encoder(global_prediction.float())[
                    :, None
                ].expand(-1, count, -1),
                structural_mean[:, None].expand(-1, count, -1),
                port_mean[:, None].expand(-1, count, -1),
                time_token[:, None].expand(-1, count, -1),
            ],
            dim=2,
        )
        flat_atoms = atom_tokens.permute(0, 2, 1, 3).reshape(
            batch * count,
            atoms,
            self.hidden,
        )
        flat_context = context.reshape(batch * count, self.context_types, self.hidden)
        allowed = self.allowed.expand(batch * count, -1, -1)
        updates = self.attention(flat_atoms, flat_context, allowed)
        flat_branch_mask = branch_mask[:, None].expand(-1, count, -1).reshape(
            batch * count,
            atoms,
        )
        no_active = ~flat_branch_mask.any(dim=-1)
        safe_branch_mask = flat_branch_mask.clone()
        safe_branch_mask[no_active, 0] = True
        atom_edges = self.atom_edges.expand(batch * count, -1, -1).clone()
        atom_edges &= safe_branch_mask[..., :, None]
        atom_edges &= safe_branch_mask[..., None, :]
        atom_updates = self.attention(flat_atoms, flat_atoms, atom_edges)
        merged_atoms = flat_atoms + updates + atom_updates
        logits = self.score(merged_atoms).squeeze(-1)
        reliability = 2.0 * torch.sigmoid(
            self.reliability(merged_atoms).squeeze(-1)
        )
        soft = torch.softmax(
            logits.masked_fill(~safe_branch_mask, float("-inf")),
            dim=-1,
        )
        soft = torch.where(
            no_active[:, None],
            torch.zeros_like(soft),
            soft,
        )
        gain = 0.5 * torch.tanh(self.mix_gate.float())
        weights = (
            (1.0 - gain) + gain * self.atom_count * soft
        ) * reliability
        weights = weights.reshape(batch, count, atoms).transpose(1, 2)
        min_positive_gate = weights.detach().masked_fill(
            ~branch_mask[..., None],
            float("inf"),
        ).amin()
        min_positive_gate = torch.where(
            branch_mask.any(dim=-1).any(),
            min_positive_gate,
            torch.zeros_like(min_positive_gate),
        )
        residual = (
            branch_residuals.float() * weights[..., None]
        ).sum(dim=1)
        return residual, {
            "atom_aggregator_gain": gain.detach(),
            "atom_aggregator_gate": weights.detach(),
            "atom_aggregator_reliability": reliability.detach().reshape(
                batch,
                count,
                atoms,
            ).transpose(1, 2),
            "atom_aggregator_residual_rms": (
                branch_residuals.float().square().mean(dim=-1).sqrt().detach()
            ),
            "atom_aggregator_active_fraction": (
                branch_mask.float().mean().detach()
            ),
            "atom_aggregator_gate_max": weights.detach().amax(),
            "atom_aggregator_gate_min_positive": min_positive_gate,
            "atom_aggregator_entropy": -(
                soft.clamp_min(1.0e-8).log() * soft
            ).sum(dim=-1).mean().detach(),
        }


class StructuredHeadSuite(nn.Module):
    """Container that keeps each structured-head enhancement independently switchable."""

    def __init__(
        self,
        hidden: int,
        *,
        hierarchy_coupling: bool = True,
        factorized_block: bool = True,
        factorized_shared_trunk: bool = False,
        factorized_rho_trust_radius: float = 0.0,
        dual_semantic_ports: bool = True,
        legalizer_response: bool = True,
    ) -> None:
        super().__init__()
        self.hierarchy_coupling = (
            MembershipHierarchyCoupler(hidden) if hierarchy_coupling else None
        )
        self.factorized_block = (
            FactorizedBlockHead(
                hidden,
                shared_trunk=factorized_shared_trunk,
                rho_trust_radius=factorized_rho_trust_radius,
            )
            if factorized_block
            else None
        )
        self.dual_semantic_ports = (
            DualSemanticPortHead(hidden) if dual_semantic_ports else None
        )
        self.legalizer_response = (
            LegalizerResponseHead(hidden) if legalizer_response else None
        )
        self.atom_aggregator = AtomResidualAggregator(hidden)

    @property
    def enabled(self) -> bool:
        return any(
            module is not None
            for module in (
                self.hierarchy_coupling,
                self.factorized_block,
                self.dual_semantic_ports,
                self.legalizer_response,
                self.atom_aggregator,
            )
        )

    def predict_legalizer_response(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        block_prediction: torch.Tensor,
        time_token: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.legalizer_response is None:
            return None, None
        return self.legalizer_response(
            block_tokens,
            block_state,
            block_prediction,
            time_token,
        )


__all__ = [
    "AtomResidualAggregator",
    "DualSemanticPortHead",
    "FactorizedBlockHead",
    "MembershipHierarchyCoupler",
    "StructuredHeadSuite",
]
