"""Gradient-isolated mask/class paths built from the mature room decoder."""

from __future__ import annotations

import torch
from torch import nn

from floorflow.heads.room_mask_decoder import RoomMaskDecoder


class DualPathRoomMaskDecoder(nn.Module):
    """Keep query masks and semantic classification on separate optimizers."""

    def __init__(
        self,
        in_channels: int,
        num_room_classes: int,
        hidden_dim: int = 128,
        num_queries: int = 32,
        layers: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        common = {
            "hidden_dim": hidden_dim,
            "num_queries": num_queries,
            "layers": layers,
            "heads": heads,
        }
        self.mask_branch = RoomMaskDecoder(
            in_channels,
            num_room_classes,
            **common,
        )
        self.class_branch = RoomMaskDecoder(
            in_channels,
            num_room_classes,
            **common,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_output = self.mask_branch(features)
        class_output = self.class_branch(features)
        return {
            "class_logits": class_output["class_logits"],
            "mask_logits": mask_output["mask_logits"],
            "aux_mask_logits": mask_output["aux_mask_logits"],
            "aux_class_logits": class_output["aux_class_logits"],
            "fusion_scale": mask_output["fusion_scale"],
        }


__all__ = ["DualPathRoomMaskDecoder"]
