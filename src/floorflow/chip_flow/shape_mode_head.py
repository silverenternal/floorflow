"""Bounded multi-mode rho hypotheses on a shared chip-flow endpoint."""

from __future__ import annotations

import math

import torch
from torch import nn

from floorflow.chip_flow.projections import _masked_lower_median
from floorflow.chip_flow.shape_mode_objectives import (
    ShapeModeOutput,
    mode_diversity_loss,
    mode_usage_loss,
    shape_mode_auxiliary_loss,
    shape_mode_latents_from_prediction,
    shape_mode_output_from_prediction,
    shape_mode_training_loss,
    winner_take_all_loss,
)


class ShapeModeHead(nn.Module):
    """Predict K bounded center/rho offsets from shared per-block context.

    The final projection is zero-initialized while the outer gate starts at
    one. This preserves exact identity at initialization without the
    multiplicative zero-gradient deadlock caused by two zero factors.
    """

    def __init__(
        self,
        hidden: int,
        *,
        modes: int = 4,
        width: int | None = None,
        max_delta: float = 0.5,
        center_scale: float = 0.0,
        mode_top_fractions: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or modes <= 0:
            raise ValueError("hidden and modes must be positive")
        if max_delta <= 0.0:
            raise ValueError("max_delta must be positive")
        if center_scale < 0.0:
            raise ValueError("center_scale must be non-negative")
        self.hidden = int(hidden)
        self.modes = int(modes)
        self.width = int(width or max(16, hidden // 8))
        self.max_delta = float(max_delta)
        self.center_scale = float(center_scale)
        self.block_norm = nn.LayerNorm(hidden)
        self.block_encoder = nn.Linear(hidden, self.width)
        self.state_encoder = nn.Linear(3, self.width)
        self.group_encoder = nn.Linear(hidden, self.width)
        self.global_encoder = nn.Linear(hidden, self.width)
        self.mode_embedding = nn.Parameter(
            torch.randn(self.modes, self.width) * 0.02
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(4 * self.width),
            nn.Linear(4 * self.width, 2 * self.width),
            nn.SiLU(),
            nn.Linear(2 * self.width, self.width),
            nn.SiLU(),
        )
        self.output = nn.Linear(self.width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.center_output = (
            nn.Linear(self.width, 2) if self.center_scale > 0.0 else None
        )
        if self.center_output is not None:
            nn.init.zeros_(self.center_output.weight)
            nn.init.zeros_(self.center_output.bias)
        self.mode_score = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, 1),
        )
        nn.init.normal_(self.mode_score[-1].weight, std=0.02)
        nn.init.zeros_(self.mode_score[-1].bias)
        self.mode_feature_encoder = nn.Linear(8, self.width)
        nn.init.normal_(self.mode_feature_encoder.weight, std=0.02)
        nn.init.zeros_(self.mode_feature_encoder.bias)
        mode_bias = torch.zeros(self.modes)
        mode_bias[0] = 10.0
        self.mode_bias = nn.Parameter(mode_bias)
        self.baseline_feature_score = nn.Linear(8, 1)
        nn.init.normal_(self.baseline_feature_score.weight, std=0.02)
        nn.init.zeros_(self.baseline_feature_score.bias)
        self.baseline_bias = nn.Parameter(torch.tensor(20.0))
        self.mode_top_fractions = (
            None
            if mode_top_fractions is None
            else tuple(float(value) for value in mode_top_fractions)
        )
        if self.mode_top_fractions is not None and (
            len(self.mode_top_fractions) != self.modes
            or any(
                not 0.0 < value <= 1.0
                for value in self.mode_top_fractions
            )
        ):
            raise ValueError(
                "mode_top_fractions must cover all modes in (0,1]"
            )
        self.gate = nn.Parameter(torch.ones(1))

    def forward(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        *,
        membership: torch.Tensor | None = None,
        structural_tokens: torch.Tensor | None = None,
        structural_mask: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
        mode_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bounded rho residuals and mode-selection logits."""

        residuals, _center_residuals, mode_logits = self._mode_outputs(
            block_tokens,
            block_state,
            membership=membership,
            structural_tokens=structural_tokens,
            structural_mask=structural_mask,
            global_token=global_token,
            mode_features=mode_features,
        )
        return residuals, mode_logits

    def _mode_outputs(
        self,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        *,
        membership: torch.Tensor | None = None,
        structural_tokens: torch.Tensor | None = None,
        structural_mask: torch.Tensor | None = None,
        global_token: torch.Tensor | None = None,
        mode_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the full candidate residual family before decoding."""

        if block_tokens.ndim != 3 or block_tokens.shape[-1] != self.hidden:
            raise ValueError("block_tokens must have shape (batch, blocks, hidden)")
        batch, blocks, _ = block_tokens.shape
        if block_state.shape != (batch, blocks, 3):
            raise ValueError("block_state must have shape (batch, blocks, 3)")
        block_code = self.block_encoder(self.block_norm(block_tokens.float()))
        state_code = self.state_encoder(block_state.float())
        group_code = self._group_code(
            block_tokens,
            membership,
            structural_tokens,
            structural_mask,
        )
        if global_token is None:
            global_code = block_code.new_zeros(batch, self.width)
        else:
            if global_token.shape != (batch, self.hidden):
                raise ValueError("global_token must have shape (batch, hidden)")
            global_code = self.global_encoder(global_token.float())
        shared = torch.cat(
            [
                block_code,
                state_code,
                group_code,
                global_code[:, None].expand(-1, blocks, -1),
            ],
            dim=-1,
        )
        mode_code = self.mode_embedding[None, :, None].expand(
            batch,
            self.modes,
            blocks,
            -1,
        )
        hidden = self.trunk(shared[:, None].expand(-1, self.modes, -1, -1))
        mode_hidden = hidden + mode_code
        score_hidden = mode_hidden
        if mode_features is not None:
            if mode_features.shape != (
                batch,
                self.modes,
                self.mode_feature_encoder.in_features,
            ):
                raise ValueError(
                    "mode_features must be [B,modes,feature_dim]"
                )
            score_hidden = mode_hidden + self.mode_feature_encoder(
                mode_features.float()
            )[:, :, None]
        residual = torch.tanh(self.output(mode_hidden).squeeze(-1))
        center_residual = (
            self.center_scale
            * torch.tanh(self.center_output(mode_hidden))
            if self.center_output is not None
            else mode_hidden.new_zeros((*mode_hidden.shape[:-1], 2))
        )
        mode_logits = (
            self.mode_score(score_hidden).squeeze(-1).mean(dim=-1)
            + self.mode_bias[None]
        )
        return (
            self.gate * self.max_delta * residual,
            center_residual,
            mode_logits,
        )

    def baseline_logit(self, baseline_features: torch.Tensor) -> torch.Tensor:
        """Score the unchanged baseline as candidate zero."""

        if baseline_features.shape != (baseline_features.shape[0], 8):
            raise ValueError("baseline_features must be [B,8]")
        return (
            self.baseline_bias
            + self.baseline_feature_score(
                baseline_features.float()
            ).squeeze(-1)
        )

    def _group_code(
        self,
        block_tokens: torch.Tensor,
        membership: torch.Tensor | None,
        structural_tokens: torch.Tensor | None,
        structural_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            membership is None
            or structural_tokens is None
            or structural_mask is None
        ):
            return block_tokens.new_zeros(
                block_tokens.shape[0],
                block_tokens.shape[1],
                self.width,
            )
        if membership.ndim != 3 or membership.shape[:2] != block_tokens.shape[:2]:
            raise ValueError("membership must match block tokens")
        slots = structural_tokens.shape[1]
        if structural_tokens.ndim != 3 or structural_tokens.shape[-1] != self.hidden:
            raise ValueError("structural_tokens must match hidden width")
        if membership.shape[-1] != slots or structural_mask.shape != (
            block_tokens.shape[0],
            slots,
        ):
            raise ValueError("structural context shapes do not match membership")
        weight = membership.float() * structural_mask[:, None].float()
        mass = weight.sum(dim=-1, keepdim=True)
        pooled = torch.einsum("bns,bsh->bnh", weight, structural_tokens.float())
        pooled = pooled / mass.clamp_min(1.0)
        return self.group_encoder(pooled)

    def forward_with_latents(
        self,
        block_prediction: torch.Tensor,
        block_tokens: torch.Tensor,
        block_state: torch.Tensor,
        constraints: torch.Tensor | None = None,
        mode_features: torch.Tensor | None = None,
        **context,
    ) -> ShapeModeOutput:
        if block_prediction.shape != block_state.shape:
            raise ValueError("block prediction and state shapes must match")
        residuals, center_residuals, mode_logits = self._mode_outputs(
            block_tokens,
            block_prediction,
            mode_features=mode_features,
            **context,
        )
        return ShapeModeOutput(
            rho_residuals=residuals,
            center_residuals=center_residuals,
            mode_latents=apply_shape_modes(
                block_prediction,
                residuals,
                center_residuals=center_residuals,
                constraints=constraints,
                top_fractions=self.mode_top_fractions,
            ),
            mode_logits=mode_logits,
        )


def apply_shape_modes(
    block_prediction: torch.Tensor,
    rho_residuals: torch.Tensor,
    *,
    center_residuals: torch.Tensor | None = None,
    constraints: torch.Tensor | None = None,
    top_fractions: tuple[float, ...] | None = None,
    rho_limit: float = 4.0,
) -> torch.Tensor:
    """Create exact-area-compatible latents ``(batch, modes, blocks, 3)``."""

    if block_prediction.ndim != 3 or block_prediction.shape[-1] != 3:
        raise ValueError("block_prediction must have shape (batch, blocks, 3)")
    batch, blocks, _ = block_prediction.shape
    if rho_residuals.shape != (batch, rho_residuals.shape[1], blocks):
        raise ValueError("rho residuals must have shape (batch, modes, blocks)")
    if rho_limit <= 0.0:
        raise ValueError("rho_limit must be positive")
    if center_residuals is not None and center_residuals.shape != (
        batch,
        rho_residuals.shape[1],
        blocks,
        2,
    ):
        raise ValueError("center residuals must be [B,modes,blocks,2]")
    safe_residuals = rho_residuals
    anchored = None
    mib_ids = None
    mib_residuals = None
    if constraints is not None:
        if constraints.shape != (batch, blocks, 5):
            raise ValueError("constraints must have shape (batch, blocks, 5)")
        anchored = (constraints[..., 0] != 0) | (constraints[..., 1] != 0)
        safe_residuals = torch.where(
            anchored[:, None],
            torch.zeros_like(rho_residuals),
            rho_residuals,
        )
        mib_ids = constraints[..., 2].long()
        mib_residuals = torch.zeros_like(rho_residuals)
        for batch_index in range(batch):
            for group in torch.unique(mib_ids[batch_index][mib_ids[batch_index] > 0]):
                members = mib_ids[batch_index].eq(group)
                candidate_rho = (
                    block_prediction[batch_index, members, 2][None]
                    + safe_residuals[batch_index, :, members]
                )
                anchored_members = anchored[batch_index, members]
                if bool(anchored_members.any()):
                    shared = candidate_rho[:, anchored_members].mean(dim=-1)
                else:
                    shared = _masked_lower_median(
                        candidate_rho,
                        torch.ones_like(candidate_rho, dtype=torch.bool),
                    )
                base_rho = block_prediction[batch_index, members, 2][None]
                mib_residuals[batch_index, :, members] = torch.where(
                    anchored_members[None],
                    torch.zeros_like(shared[:, None]),
                    shared[:, None] - base_rho,
                )
                safe_residuals[batch_index, :, members] = torch.where(
                    anchored_members[None],
                    torch.zeros_like(shared[:, None]),
                    shared[:, None] - base_rho,
                )
    if top_fractions is not None:
        if len(top_fractions) != safe_residuals.shape[1] or any(
            not 0.0 < float(value) <= 1.0
            for value in top_fractions
        ):
            raise ValueError("top_fractions must cover every mode in (0,1]")
        for mode, fraction in enumerate(top_fractions):
            count = max(1, math.ceil(float(fraction) * blocks))
            magnitude = safe_residuals[:, mode].abs()
            threshold = magnitude.topk(count, dim=1).values[:, -1:]
            selected = magnitude.ge(threshold)
            safe_residuals[:, mode] = torch.where(
                selected,
                safe_residuals[:, mode],
                torch.zeros_like(safe_residuals[:, mode]),
            )
        if constraints is not None and mib_ids is not None and mib_residuals is not None:
            for batch_index in range(batch):
                for group in torch.unique(
                    mib_ids[batch_index][mib_ids[batch_index] > 0]
                ):
                    members = mib_ids[batch_index].eq(group)
                    free = members & ~anchored[batch_index]
                    any_selected = safe_residuals[
                        batch_index,
                        :,
                        free,
                    ].ne(0).any(dim=-1)
                    shared = torch.where(
                        any_selected[:, None],
                        mib_residuals[batch_index][:, free],
                        torch.zeros_like(mib_residuals[batch_index][:, free]),
                    )
                    safe_residuals[batch_index, :, free] = shared
    xy = block_prediction[:, None, :, :2].expand(
        -1,
        rho_residuals.shape[1],
        -1,
        -1,
    )
    if center_residuals is not None:
        safe_centers = center_residuals
        if constraints is not None:
            preplaced = constraints[..., 1].ne(0)
            safe_centers = torch.where(
                preplaced[:, None, :, None],
                torch.zeros_like(center_residuals),
                center_residuals,
            )
        xy = xy + safe_centers
    rho = (
        block_prediction[:, None, :, 2] + safe_residuals
    ).clamp(-rho_limit, rho_limit)
    return torch.cat([xy, rho[..., None]], dim=-1)


def attach_shape_mode_head(
    model: nn.Module,
    *,
    modes: int = 4,
    width: int | None = None,
    max_delta: float = 0.5,
    center_scale: float = 0.0,
    apply: bool = False,
    selection: str = "learned",
    mode_top_fractions: tuple[float, ...] | None = None,
) -> nn.Module:
    """Attach the head module without adding per-step forward work."""

    hidden = int(getattr(model, "hidden", 0))
    if hidden <= 0:
        raise ValueError("shape-mode adapter requires a hidden model width")
    if getattr(model, "shape_mode_head", None) is not None:
        return model
    head = ShapeModeHead(
        hidden,
        modes=modes,
        width=width,
        max_delta=max_delta,
        center_scale=center_scale,
        mode_top_fractions=mode_top_fractions,
    )
    model.add_module("shape_mode_head", head)
    model.shape_mode_apply = bool(apply)
    if selection not in {"learned", "pareto"}:
        raise ValueError("shape-mode selection must be learned or pareto")
    model.shape_mode_selection = selection

    return model


__all__ = [
    "ShapeModeHead",
    "ShapeModeOutput",
    "apply_shape_modes",
    "attach_shape_mode_head",
    "mode_diversity_loss",
    "mode_usage_loss",
    "shape_mode_auxiliary_loss",
    "shape_mode_latents_from_prediction",
    "shape_mode_output_from_prediction",
    "shape_mode_training_loss",
    "winner_take_all_loss",
]
