"""Granular train scopes for the decomposed opening head."""

from __future__ import annotations

import torch

SCOPES = (
    "full",
    "foreground",
    "center",
    "width",
    "family",
    "width_family",
    "center_width",
    "gap_fusion",
    "output_heads",
)


def _matches(name: str, scope: str) -> bool:
    if scope == "full":
        return True
    if scope == "foreground":
        return name.startswith("foreground.")
    if scope == "center":
        return name.startswith("center.")
    if scope == "width":
        return name.startswith("width.")
    if scope == "family":
        return name.startswith("family.")
    if scope == "width_family":
        return name.startswith(("width.", "family.", "family_context."))
    if scope == "center_width":
        return name.startswith(("center.", "width."))
    if scope == "gap_fusion":
        return name.startswith(("gap_",))
    if scope == "output_heads":
        return name.startswith(("foreground.", "center.", "width.", "family."))
    raise ValueError(f"unknown opening train scope: {scope}")


def configure_opening_head_scope(
    head: torch.nn.Module,
    scope: str,
) -> dict[str, object]:
    """Freeze all opening-head parameters outside one gap-specific scope."""

    if scope not in SCOPES:
        raise ValueError(f"unknown opening train scope: {scope}")
    selected = []
    frozen = 0
    for name, parameter in head.named_parameters():
        trainable = _matches(name, scope)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(name)
        else:
            frozen += parameter.numel()
    if not selected:
        raise RuntimeError(f"opening train scope {scope!r} selected no parameters")
    trainable_count = sum(
        parameter.numel()
        for parameter in head.parameters()
        if parameter.requires_grad
    )
    return {
        "scope": scope,
        "parameters": selected,
        "trainable_count": trainable_count,
        "frozen_count": frozen,
    }


__all__ = ["SCOPES", "configure_opening_head_scope"]
