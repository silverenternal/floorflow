"""Parameter subsets for expensive GPU gradient procedures."""

from __future__ import annotations

TOKEN_HEAD_PREFIXES = (
    "token_head.",
    "token_memory.",
    "token_refiner.",
    "token_graph_head.",
    "codetree_head.",
    "room_instance_head.",
    "query_pool.",
)


def scoped_parameters(model, scope: str):
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if scope == "all":
        return trainable
    if scope == "token_heads":
        selected = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(TOKEN_HEAD_PREFIXES)
        ]
        return selected or trainable
    raise ValueError(f"unknown parameter scope: {scope}")


__all__ = ["TOKEN_HEAD_PREFIXES", "scoped_parameters"]
