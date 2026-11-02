"""Parameter-group construction for token and backbone optimization."""

from __future__ import annotations

import torch
from torch import nn

TOKEN_PARAMETER_PREFIXES = (
    "token_head.",
    "query_pool.",
    "token_memory.",
    "token_refiner.",
)


def build_floorflow_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    token_learning_rate_multiplier: float = 1.0,
) -> torch.optim.AdamW:
    if learning_rate <= 0.0 or token_learning_rate_multiplier <= 0.0:
        raise ValueError("learning rates must be positive")
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_token = name.startswith(TOKEN_PARAMETER_PREFIXES)
        no_decay = (
            name.endswith("bias")
            or ".norm" in name
            or name.endswith(".weight")
            and (
                "embedding" in name
                or ".token." in name
                or ".position." in name
                or ".stage." in name
            )
        )
        groups.setdefault((is_token, no_decay), []).append(parameter)
    parameters = []
    for (is_token, no_decay), values in sorted(groups.items()):
        parameters.append(
            {
                "params": values,
                "lr": learning_rate
                * (token_learning_rate_multiplier if is_token else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": (
                    f"{'token' if is_token else 'backbone'}_"
                    f"{'no_decay' if no_decay else 'decay'}"
                ),
            }
        )
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(parameters)


__all__ = ["TOKEN_PARAMETER_PREFIXES", "build_floorflow_optimizer"]
