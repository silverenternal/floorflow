"""Checkpoint-safe integration of learned pixel-distribution gates."""

from __future__ import annotations

from pathlib import Path

import torch

from floorflow.heads.pixel_distribution import PixelDistributionHead
from floorflow.model.distribution_controller import PixelDistributionController
from floorflow.model.sparse_control_attention import MaskedControlFlow


def load_distribution_head_state(
    path: str | Path,
    in_channels: int,
    num_classes: int,
    hidden_channels: int,
    room_ids: tuple[int, ...],
) -> tuple[PixelDistributionHead, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    head = PixelDistributionHead(
        int(in_channels),
        int(num_classes),
        hidden_channels=int(hidden_channels),
        room_ids=room_ids,
    )
    head.load_state_dict(state)
    return head, payload


def attach_distribution_controller(
    model,
    taxonomy,
    *,
    head_state=None,
    gate_state=None,
    hidden_channels: int = 96,
    gate_hidden: int = 64,
):
    """Attach a trainable distribution head/gate without changing old models."""

    if head_state is None and gate_state is None:
        return model
    head = PixelDistributionHead(
        int(model.output.in_channels),
        taxonomy.num_classes,
        hidden_channels=int(hidden_channels),
        room_ids=taxonomy.room_ids,
    )
    if head_state is not None:
        head.load_state_dict(head_state)
    controller = PixelDistributionController(
        model,
        num_classes=taxonomy.num_classes,
        room_ids=taxonomy.room_ids,
        hidden_channels=int(hidden_channels),
        distribution_head=head,
        velocity_residual=True,
        gate_hidden=int(gate_hidden),
    )
    if gate_state is not None:
        controller.velocity_gate.load_state_dict(gate_state)
    return controller


def unwrap_checkpoint_models(model):
    """Return core model, distribution controller, and sparse wrapper if present."""

    sparse = model if isinstance(model, MaskedControlFlow) else None
    core = sparse.base if sparse is not None else model
    controller = (
        core if isinstance(core, PixelDistributionController) else None
    )
    core_model = controller.base if controller is not None else core
    return core_model, controller, sparse


def distribution_checkpoint_fields(model) -> dict[str, object]:
    core, controller, sparse = unwrap_checkpoint_models(model)
    fields: dict[str, object] = {"model": core.state_dict()}
    if controller is not None:
        fields["distribution_head"] = controller.distribution.state_dict()
        if controller.velocity_gate is not None:
            fields["distribution_gate"] = controller.velocity_gate.state_dict()
    if sparse is not None:
        fields["sparse_attention"] = sparse.sparse_attention.state_dict()
    return fields


__all__ = [
    "attach_distribution_controller",
    "distribution_checkpoint_fields",
    "load_distribution_head_state",
    "unwrap_checkpoint_models",
]
