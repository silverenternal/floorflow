"""Typed, provenance-preserving inputs for CAD inverse generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ConditionMode = Literal[
    "contour_only",
    "sparse_control",
    "layout_conditioned",
]


@dataclass(frozen=True)
class ConditionView:
    """Split model-visible controls from hidden target-derived controls.

    ``control`` may only contain values visible under ``visibility``. Hidden
    target values are therefore impossible to recover through this object.
    """

    mode: ConditionMode
    footprint: torch.Tensor
    control: torch.Tensor
    visibility: torch.Tensor
    floor_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode not in {
            "contour_only",
            "sparse_control",
            "layout_conditioned",
        }:
            raise ValueError(f"unknown condition mode: {self.mode}")
        if self.footprint.ndim == 3:
            object.__setattr__(self, "footprint", self.footprint[:, None])
        if self.visibility.ndim == 3:
            object.__setattr__(self, "visibility", self.visibility[:, None])
        if self.control.ndim != 4:
            raise ValueError("control must be [B,C,H,W]")
        if self.control.shape[0] != self.footprint.shape[0]:
            raise ValueError("batch sizes do not match")
        if self.control.shape[-2:] != self.footprint.shape[-2:]:
            raise ValueError("control and footprint spatial sizes do not match")
        if self.visibility.shape[0] != self.footprint.shape[0]:
            raise ValueError("visibility batch size does not match")
        if self.visibility.shape[-2:] != self.footprint.shape[-2:]:
            raise ValueError("visibility and footprint spatial sizes do not match")
        if self.visibility.shape[1] not in {1, self.control.shape[1]}:
            raise ValueError("visibility must have one or all control channels")
        if self.visibility.dtype != torch.bool:
            raise ValueError("visibility must be boolean")
        if self.floor_ids is not None:
            if self.floor_ids.ndim != 1:
                raise ValueError("floor_ids must be [B]")
            if self.floor_ids.shape[0] != self.footprint.shape[0]:
                raise ValueError("floor_ids batch size does not match")
        inside = self.footprint.bool().expand_as(self.visibility)
        if bool((self.visibility & ~inside).any()):
            raise ValueError("visibility cannot include pixels outside the footprint")
        if self.mode == "contour_only":
            if bool(self.visibility.any()) or bool((self.control != 0).any()):
                raise ValueError("contour_only cannot expose coarse controls")
        elif self.mode == "layout_conditioned":
            if not torch.equal(
                self.visibility.expand_as(self.control),
                self.footprint.bool().expand_as(self.control),
            ):
                raise ValueError(
                    "layout_conditioned must expose the full interior footprint"
                )
        if bool((self.control != 0).any()):
            visible = self.visibility.expand_as(self.control)
            if bool((self.control[~visible] != 0).any()):
                raise ValueError("hidden controls must be exactly zero")

    @property
    def hidden_mask(self) -> torch.Tensor:
        """Return the spatially expanded mask of unavailable controls."""

        inside = self.footprint.bool().expand_as(self.control)
        return inside & ~self.visibility.expand_as(self.control)

    @property
    def is_target_control_free(self) -> bool:
        return not bool(self.visibility.any())

    def manifest(self) -> dict[str, object]:
        """Return machine-checkable evidence for an evaluation manifest."""

        visible = self.visibility.expand_as(self.control)
        return {
            "mode": self.mode,
            "batch_size": int(self.control.shape[0]),
            "channels": int(self.control.shape[1]),
            "spatial_shape": list(self.control.shape[-2:]),
            "footprint_fraction": float(self.footprint.float().mean().cpu()),
            "visible_control_fraction": float(
                visible.float().mean().cpu()
            ),
            "target_control_free": self.is_target_control_free,
            "floor_ids_present": self.floor_ids is not None,
        }


def contour_only_view(
    footprint: torch.Tensor,
    control_channels: int,
    *,
    floor_ids: torch.Tensor | None = None,
) -> ConditionView:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    control = torch.zeros(
        footprint.shape[0],
        int(control_channels),
        *footprint.shape[-2:],
        device=footprint.device,
        dtype=footprint.dtype,
    )
    visibility = torch.zeros_like(footprint, dtype=torch.bool)
    return ConditionView("contour_only", footprint, control, visibility, floor_ids)


def sparse_control_view(
    control: torch.Tensor,
    footprint: torch.Tensor,
    hidden_mask: torch.Tensor,
    *,
    floor_ids: torch.Tensor | None = None,
) -> ConditionView:
    """Keep only controls outside ``hidden_mask``.

    This factory is the only supported way to construct a sparse view. It
    discards hidden values before they can reach a mask generator or model.
    """

    if footprint.ndim == 3:
        footprint = footprint[:, None]
    if hidden_mask.ndim == 3:
        hidden_mask = hidden_mask[:, None]
    if hidden_mask.shape[-2:] != footprint.shape[-2:]:
        raise ValueError("hidden_mask spatial shape does not match footprint")
    if hidden_mask.shape[0] != footprint.shape[0]:
        raise ValueError("hidden_mask batch size does not match footprint")
    if hidden_mask.shape[1] not in {1, control.shape[1]}:
        raise ValueError("hidden_mask must have one or all control channels")
    visibility = footprint.bool() & ~hidden_mask.bool()
    visible_control = torch.where(
        visibility.expand_as(control),
        control,
        torch.zeros_like(control),
    )
    return ConditionView(
        "sparse_control",
        footprint,
        visible_control,
        visibility,
        floor_ids,
    )


def layout_conditioned_view(
    control: torch.Tensor,
    footprint: torch.Tensor,
    *,
    floor_ids: torch.Tensor | None = None,
) -> ConditionView:
    if footprint.ndim == 3:
        footprint = footprint[:, None]
    visibility = footprint.bool().expand_as(control)
    visible_control = torch.where(visibility, control, torch.zeros_like(control))
    return ConditionView(
        "layout_conditioned",
        footprint,
        visible_control,
        footprint.bool(),
        floor_ids,
    )


__all__ = [
    "ConditionMode",
    "ConditionView",
    "contour_only_view",
    "layout_conditioned_view",
    "sparse_control_view",
]
