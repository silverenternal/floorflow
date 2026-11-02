"""Anchor-preserving low-rank adapters for the pinned chip backbone."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Linear):
    """Linear layer with a frozen base weight and an additive low-rank delta."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        if rank <= 0 or alpha <= 0.0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.weight = nn.Parameter(
            base.weight.detach().clone(),
            requires_grad=False,
        )
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                base.bias.detach().clone(),
                requires_grad=False,
            )
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = super().forward(values)
        delta = F.linear(F.linear(values, self.lora_a), self.lora_b)
        return base + self.scale * delta


def install_backbone_lora(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
) -> None:
    """Replace selected block linears while preserving their base state keys."""

    targets = []
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("blocks."):
            continue
        if not name.endswith(
            ("attn.qkv", "attn.proj", "mlp.w_in", "mlp.w_out")
        ):
            continue
        targets.append((name, module))
    if not targets:
        raise ValueError("no backbone LoRA target linears found")
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha),
        )
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(name.endswith(("lora_a", "lora_b")))


__all__ = ["LoRALinear", "install_backbone_lora"]
