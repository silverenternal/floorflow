"""Structured decoding, repair, selection, and sampling policies."""

from __future__ import annotations

__all__ = ["semantic_endpoint_labels"]


def __getattr__(name: str):
    if name == "semantic_endpoint_labels":
        from .core import semantic_endpoint_labels

        return semantic_endpoint_labels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
