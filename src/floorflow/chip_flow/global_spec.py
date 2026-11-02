"""Target-free global design geometry conditioning."""

from __future__ import annotations

import torch
from torch import nn

GLOBAL_SPEC_DIM = 7
GLOBAL_SPEC_FEATURES = (
    "core_width_over_area_scale",
    "core_height_over_area_scale",
    "log_core_aspect",
    "die_width_over_core_width",
    "die_height_over_core_height",
    "macro_utilization",
    "log_core_area_over_macro_area",
)


def global_spec_vector(
    *,
    core_width: torch.Tensor,
    core_height: torch.Tensor,
    die_width: torch.Tensor,
    die_height: torch.Tensor,
    macro_area: torch.Tensor,
) -> torch.Tensor:
    """Build normalized global geometry from supplied design inputs only."""

    values = torch.broadcast_tensors(
        core_width.float(),
        core_height.float(),
        die_width.float(),
        die_height.float(),
        macro_area.float(),
    )
    core_width, core_height, die_width, die_height, macro_area = values
    if not torch.isfinite(
        torch.stack(
            [core_width, core_height, die_width, die_height, macro_area],
            dim=-1,
        )
    ).all():
        raise ValueError("global design geometry and area must be finite")
    if torch.any(core_width <= 0.0) or torch.any(core_height <= 0.0):
        raise ValueError("core dimensions must be positive")
    if torch.any(die_width <= 0.0) or torch.any(die_height <= 0.0):
        raise ValueError("die dimensions must be positive")
    if torch.any(macro_area <= 0.0):
        raise ValueError("macro area must be positive")
    if torch.any(die_width < core_width) or torch.any(die_height < core_height):
        raise ValueError("die dimensions must contain the core dimensions")
    area_scale = macro_area.sqrt().clamp_min(1.0e-6)
    utilization = (macro_area / (core_width * core_height).clamp_min(1.0e-6)).clamp(
        0.0,
        1.0,
    )
    return torch.stack(
        [
            core_width / area_scale,
            core_height / area_scale,
            torch.log(core_width / core_height),
            die_width / core_width,
            die_height / core_height,
            utilization,
            torch.log((core_width * core_height) / macro_area),
        ],
        dim=-1,
    )


class GlobalSpecConditioner(nn.Module):
    """Zero-gated embedding of an explicit global design spec."""

    def __init__(
        self,
        hidden: int,
        *,
        input_dim: int = GLOBAL_SPEC_DIM,
        width: int | None = None,
    ) -> None:
        super().__init__()
        if hidden <= 0 or input_dim <= 0:
            raise ValueError("hidden and input_dim must be positive")
        self.hidden = int(hidden)
        self.input_dim = int(input_dim)
        bottleneck = int(width or max(16, hidden // 8))
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        spec: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if spec.ndim != 2 or spec.shape[-1] != self.input_dim:
            raise ValueError(
                f"global spec must have shape (batch, {self.input_dim})"
            )
        if mask is None:
            mask = spec.new_ones(spec.shape[0])
        if mask.shape != (spec.shape[0],):
            raise ValueError("global-spec mask must have shape (batch,)")
        valid = mask.bool()
        safe_spec = torch.where(
            valid[:, None],
            spec.float(),
            spec.new_zeros(()),
        )
        condition = self.gate * self.network(safe_spec)
        return torch.where(valid[:, None], condition, condition.new_zeros(()))


__all__ = [
    "GLOBAL_SPEC_DIM",
    "GLOBAL_SPEC_FEATURES",
    "GlobalSpecConditioner",
    "global_spec_vector",
]
