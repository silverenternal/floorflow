"""Objective composition and gradient-contribution diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ObjectiveTerm:
    """A differentiable objective term and its explicit scalar weight."""

    name: str
    value: torch.Tensor
    weight: float = 1.0

    @property
    def weighted_value(self) -> torch.Tensor:
        return float(self.weight) * self.value


def compose_objective(terms: Iterable[ObjectiveTerm]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the weighted sum and detached component values."""

    materialized = tuple(terms)
    if not materialized:
        raise ValueError("at least one objective term is required")
    total = sum((term.weighted_value for term in materialized[1:]), materialized[0].weighted_value)
    components = {term.name: term.value.detach() for term in materialized}
    components["loss"] = total.detach()
    return total, components


def contribution_rows(
    terms: Iterable[ObjectiveTerm],
    *,
    prefix: str = "train",
) -> list[dict[str, float | str]]:
    """Report raw, weighted, signed, and fractional loss mass for every term."""

    materialized = tuple(terms)
    signed_total = float(
        sum(term.weighted_value.detach().cpu() for term in materialized)
    )
    absolute_total = float(
        sum(abs(term.weighted_value.detach().cpu()) for term in materialized)
    )
    rows: list[dict[str, float | str]] = []
    for term in materialized:
        weighted = float(term.weighted_value.detach().cpu())
        rows.append(
            {
                "split": prefix,
                "term": term.name,
                "raw": float(term.value.detach().cpu()),
                "weight": float(term.weight),
                "weighted": weighted,
                "signed_share": weighted / signed_total if signed_total != 0.0 else 0.0,
                "absolute_share": abs(weighted) / absolute_total if absolute_total > 0.0 else 0.0,
            }
        )
    return rows


def _flat_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    if not loss.requires_grad:
        return tuple(torch.zeros_like(parameter) for parameter in parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        materialize_grads=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )


def gradient_contribution_rows(
    terms: Iterable[ObjectiveTerm],
    parameters: Iterable[torch.nn.Parameter],
    *,
    primary: str,
) -> list[dict[str, float | str]]:
    """Measure per-term gradient norm and cosine conflict against the primary term.

    This calls ``autograd.grad`` with a retained graph and is intended for sparse
    diagnostics rather than every optimizer step.
    """

    materialized = tuple(terms)
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    named = {term.name: term for term in materialized}
    if primary not in named:
        raise ValueError(f"primary objective {primary!r} is not present")
    reference = _flat_gradients(named[primary].weighted_value, trainable, retain_graph=True)
    reference_norm = math.sqrt(sum(float(gradient.pow(2).sum().cpu()) for gradient in reference))
    rows: list[dict[str, float | str]] = []
    for term in materialized:
        gradients = _flat_gradients(
            term.weighted_value,
            trainable,
            retain_graph=True,
        )
        norm = math.sqrt(sum(float(gradient.pow(2).sum().cpu()) for gradient in gradients))
        dot = sum(float((left * right).sum().cpu()) for left, right in zip(reference, gradients))
        cosine = dot / max(reference_norm * norm, 1.0e-12)
        rows.append(
            {
                "term": term.name,
                "gradient_norm": norm,
                "cosine_with_primary": cosine,
                "gradient_ratio_to_primary": norm / max(reference_norm, 1.0e-12),
            }
        )
    return rows
