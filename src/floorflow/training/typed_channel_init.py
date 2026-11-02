"""Checkpoint initialization for typed-channel mechanism screens."""

from __future__ import annotations

from pathlib import Path

import torch


def load_typed_channel_initialization(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Load compatible weights and expose missing/unexpected state explicitly."""

    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("initialization checkpoint has no model state dict")
    result = model.load_state_dict(state, strict=False)
    stage = getattr(model, "room_instance_stage", None)
    query_head = getattr(stage, "query_set_head", None)
    reapplied_class_count_prior = bool(
        query_head is not None
        and query_head.class_count_prior.numel() > 0
    )
    if query_head is not None:
        query_head.reset_class_count_bias()
    reset_graph_residuals = bool(
        stage is not None
        and getattr(stage, "count_aware_second_graph", False)
        and getattr(stage, "room_graph_conditioner", None) is not None
    )
    if reset_graph_residuals:
        stage.room_graph_conditioner.reset_graph_residual_heads()
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "reapplied_class_count_prior": reapplied_class_count_prior,
        "reset_graph_residual_heads": reset_graph_residuals,
    }


__all__ = ["load_typed_channel_initialization"]
