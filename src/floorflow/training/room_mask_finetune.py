"""Fine-tuning scopes for the room query-mask decoder."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

CLASS_PATH_PREFIXES = ("decoder_norm.", "class_fusion.", "class_head.")
CLASSIFIER_PATH_PREFIXES = ("class_fusion.", "class_head.")
TRAIN_SCOPES = ("full", "class_only", "classifier_only")


def configure_room_mask_train_scope(
    decoder: nn.Module,
    scope: str,
) -> dict[str, object]:
    """Freeze the mask-bearing path while leaving the semantic readout trainable."""

    if scope not in TRAIN_SCOPES:
        raise ValueError(f"unsupported train scope: {scope}")
    decoder.requires_grad_(scope == "full")
    if scope == "class_only":
        for name, parameter in decoder.named_parameters():
            parameter.requires_grad_(name.startswith(CLASS_PATH_PREFIXES))
    elif scope == "classifier_only":
        for name, parameter in decoder.named_parameters():
            parameter.requires_grad_(name.startswith(CLASSIFIER_PATH_PREFIXES))
    trainable = [
        name
        for name, parameter in decoder.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError(f"train scope {scope!r} selected no parameters")
    frozen = [
        name
        for name, parameter in decoder.named_parameters()
        if not parameter.requires_grad
    ]
    return {
        "scope": scope,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_count": sum(
            parameter.numel()
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ),
        "frozen_count": sum(
            parameter.numel()
            for parameter in decoder.parameters()
            if not parameter.requires_grad
        ),
    }


def trainable_parameters(
    parameters: Iterable[nn.Parameter],
) -> list[nn.Parameter]:
    """Return the non-empty trainable parameter set for an optimizer."""

    selected = [parameter for parameter in parameters if parameter.requires_grad]
    if not selected:
        raise RuntimeError("optimizer received no trainable parameters")
    return selected


def load_room_mask_decoder_state(
    decoder: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Load legacy checkpoints while requiring explicit class-adapter compatibility."""

    missing, unexpected = decoder.load_state_dict(state, strict=False)
    allowed_missing = [
        name for name in missing if name.startswith("class_fusion.")
    ]
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(
            f"decoder state mismatch: missing={missing}, unexpected={unexpected}"
        )
    if missing:
        parameters = dict(decoder.named_parameters())
        with torch.no_grad():
            for name in missing:
                parameters[name].zero_()


@torch.no_grad()
def parameter_snapshot(
    decoder: nn.Module,
    trainable: bool,
) -> dict[str, torch.Tensor]:
    """Clone selected parameters for deterministic freeze regression tests."""

    return {
        name: parameter.detach().clone()
        for name, parameter in decoder.named_parameters()
        if parameter.requires_grad == trainable
    }


__all__ = [
    "CLASSIFIER_PATH_PREFIXES",
    "CLASS_PATH_PREFIXES",
    "TRAIN_SCOPES",
    "configure_room_mask_train_scope",
    "load_room_mask_decoder_state",
    "parameter_snapshot",
    "trainable_parameters",
]
