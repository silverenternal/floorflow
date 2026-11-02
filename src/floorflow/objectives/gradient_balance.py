"""Gradient diagnostics for multi-objective CAD training."""

from __future__ import annotations

import itertools

import torch


def gradient_diagnostics(
    losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> dict[str, object]:
    """Return gradient norms and pairwise cosines at one training step."""

    vectors: dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        flat = torch.cat(
            [
                (
                    torch.zeros_like(parameter).flatten()
                    if gradient is None
                    else gradient.detach().flatten()
                )
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
        vectors[name] = flat.float()
    norms = {
        name: float(vector.norm().cpu())
        for name, vector in vectors.items()
    }
    cosines = {}
    for left, right in itertools.combinations(vectors, 2):
        denominator = vectors[left].norm() * vectors[right].norm()
        cosine = (
            torch.dot(vectors[left], vectors[right]) / denominator
            if float(denominator) > 0
            else torch.zeros((), device=denominator.device)
        )
        cosines[f"{left}__{right}"] = float(cosine.cpu())
    return {"norms": norms, "pairwise_cosine": cosines}


__all__ = ["gradient_diagnostics"]
