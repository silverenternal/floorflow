"""Readout adapters for dense and query room controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.heads.room_mask_fusion import (
    fused_room_mask_logits,
    mask_region_class_logits,
)


def _blend_room_instance_probabilities(
    outputs: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    dense = outputs["dense_logits"].softmax(dim=1)
    fused = outputs["fused_logits"].softmax(dim=1)
    return (1.0 - float(weight)) * dense + float(weight) * fused


class RasterInstanceRoomAdapter(nn.Module):
    """Expose the selected dense/query blend as a room-logit model."""

    def __init__(self, model: nn.Module, blend: float) -> None:
        super().__init__()
        self.model = model
        self.blend = float(blend)

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(footprint, control_image=control_image)
        probability = _blend_room_instance_probabilities(outputs, self.blend)
        return probability.clamp_min(1.0e-8).log()


class RasterDenseQueryRoomBlend(nn.Module):
    """Blend independent dense and query-room checkpoints on image evidence."""

    def __init__(
        self,
        dense_model: nn.Module,
        query_model: nn.Module,
        blend: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(blend) <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self.dense_model = dense_model
        self.query_model = query_model
        self.blend = float(blend)

    def forward_with_dense(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dense_logits = self.dense_model(
            footprint,
            control_image=control_image,
        )
        query_logits = self.query_model(
            footprint,
            control_image=control_image,
        )["fused_logits"]
        if dense_logits.shape != query_logits.shape:
            raise ValueError("dense and query room logits must share a shape")
        probability = (
            (1.0 - self.blend) * dense_logits.softmax(dim=1)
            + self.blend * query_logits.softmax(dim=1)
        )
        return probability.clamp_min(1.0e-8).log(), dense_logits

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_dense(
            footprint,
            control_image=control_image,
        )[0]


class RasterInstanceClassRoutedAdapter(nn.Module):
    """Use query masks as regions while dense logits remain the class source."""

    def __init__(
        self,
        model: nn.Module,
        *,
        threshold: float = 0.5,
        strength: float = 8.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.threshold = float(threshold)
        self.strength = float(strength)

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_dense(
            footprint,
            control_image=control_image,
        )[0]

    def forward_with_dense(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(footprint, control_image=control_image)
        dense_logits = outputs["dense_logits"]
        predictions = dict(outputs)
        predictions["mask_logits"] = F.interpolate(
            outputs["mask_logits"],
            size=dense_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        routed = mask_region_class_logits(
            predictions,
            dense_logits,
            footprint,
            threshold=self.threshold,
            strength=self.strength,
        )
        return routed, dense_logits


class RasterInstanceSoftRoutedAdapter(nn.Module):
    """Blend external classes and query classes with a per-pixel support gate."""

    def __init__(
        self,
        model: nn.Module,
        *,
        threshold: float = 0.5,
        strength: float = 8.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.threshold = float(threshold)
        self.strength = float(strength)

    def forward(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_dense(
            footprint,
            control_image=control_image,
        )[0]

    def forward_with_dense(
        self,
        footprint: torch.Tensor,
        control_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(footprint, control_image=control_image)
        dense_logits = outputs["dense_logits"]
        instances = dict(outputs)
        instances["mask_logits"] = F.interpolate(
            outputs["mask_logits"],
            size=dense_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        query_logits = fused_room_mask_logits(
            instances,
            dense_logits.shape[1],
        )
        support = instances["mask_logits"].sigmoid().max(dim=1).values
        gate = torch.sigmoid(self.strength * (support - self.threshold))
        footprint_mask = (
            footprint[:, 0] if footprint.ndim == 4 else footprint
        )
        gate = gate * footprint_mask.float()
        probability = (
            (1.0 - gate[:, None]) * dense_logits.softmax(dim=1)
            + gate[:, None] * query_logits.softmax(dim=1)
        )
        return probability.clamp_min(1.0e-8).log(), dense_logits


__all__ = [
    "RasterDenseQueryRoomBlend",
    "RasterInstanceClassRoutedAdapter",
    "RasterInstanceRoomAdapter",
    "RasterInstanceSoftRoutedAdapter",
]
