"""Project coupled distributions into a contour-conditioned CFM source."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from floorflow.distribution.coupling_loss import element_to_pixel_probability
from floorflow.distribution.hierarchy import ChannelHierarchy


class DistributionCFMBridge(nn.Module):
    """Build a CFM initial state from pixel and element distributions."""

    def __init__(
        self,
        hierarchy: ChannelHierarchy,
        *,
        element_blend_init: float = 0.0,
        bridge_mode: str = "blend",
    ) -> None:
        super().__init__()
        if bridge_mode not in {"blend", "residual"}:
            raise ValueError("bridge_mode must be blend or residual")
        self.hierarchy = hierarchy
        self.bridge_mode = bridge_mode
        self.element_blend = nn.Parameter(
            torch.full(
                (hierarchy.num_classes - 1,),
                float(element_blend_init),
            )
        )
        self.element_residual = nn.Parameter(
            torch.zeros(hierarchy.num_classes - 1)
        )

    def forward(
        self,
        pixel_output: Mapping[str, torch.Tensor],
        element_output: Mapping[str, dict[str, torch.Tensor]],
        footprint: torch.Tensor,
        element_probability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if footprint.ndim == 3:
            footprint = footprint[:, None]
        pixel_state = pixel_output["probability"][:, 1:].float()
        if element_probability is None:
            element_probability = element_to_pixel_probability(
                element_output,
                self.hierarchy,
                pixel_output["probability"],
            )
        if self.bridge_mode == "residual":
            residual = torch.tanh(self.element_residual).to(pixel_state.dtype)[
                None,
                :,
                None,
                None,
            ]
            state = pixel_state + residual * element_probability
            blend = residual
        else:
            blend = torch.tanh(self.element_blend).to(pixel_state.dtype)[
                None,
                :,
                None,
                None,
            ]
            state = (
                (1.0 - blend) * pixel_state
                + blend * element_probability
            )
        state = state * footprint.float()
        return {
            "state": state,
            "pixel_state": pixel_state * footprint.float(),
            "element_state": element_probability * footprint.float(),
            "element_blend": blend.mean(),
            "element_blend_by_channel": blend.flatten(),
            "bridge_mode": (
                pixel_state.new_tensor(1.0)
                if self.bridge_mode == "residual"
                else pixel_state.new_tensor(0.0)
            ),
        }


__all__ = ["DistributionCFMBridge"]
