"""Conflict-aware fusion of hard feasibility proxies into flow refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from floorflow.objectives.soft_graph_edit import soft_graph_edit_distance


@dataclass(frozen=True)
class FusionDiagnostics:
    mode: str
    base_norm: float
    constraint_norm: float
    cosine: float
    projected_constraint_norm: float
    projected_fraction: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )


def _dot(left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(
        (left_gradient * right_gradient).sum()
        for left_gradient, right_gradient in zip(left, right)
    )


def _norm(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.sqrt(
        sum(gradient.pow(2).sum() for gradient in gradients)
    )


def apply_constraint_fusion(
    base_loss: torch.Tensor,
    constraint_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    mode: str = "projection",
    constraint_scale: float = 0.05,
) -> FusionDiagnostics:
    """Populate gradients using base, scalar, or conflict-projected fusion."""

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if mode == "base":
        base_loss.backward()
        return FusionDiagnostics("base", 0.0, 0.0, 0.0, 0.0, 0.0)
    if mode not in {"scalar", "projection"}:
        raise ValueError(f"unknown fusion mode: {mode}")
    base_gradients = _gradients(base_loss, trainable, retain_graph=True)
    constraint_gradients = _gradients(
        constraint_loss,
        trainable,
        retain_graph=False,
    )
    base_norm = _norm(base_gradients)
    constraint_norm = _norm(constraint_gradients)
    dot = _dot(base_gradients, constraint_gradients)
    cosine = dot / base_norm.clamp_min(1.0e-12) / constraint_norm.clamp_min(1.0e-12)
    projected = constraint_gradients
    if mode == "projection" and float(cosine.detach().cpu()) < 0.0:
        projected = tuple(
            constraint_gradient
            - (
                dot
                / base_norm.pow(2).clamp_min(1.0e-12)
                * base_gradient
            )
            for base_gradient, constraint_gradient in zip(
                base_gradients,
                constraint_gradients,
            )
        )
    projected_norm = _norm(projected)
    for parameter, base_gradient, constraint_gradient in zip(
        trainable,
        base_gradients,
        projected,
    ):
        parameter.grad = base_gradient + constraint_scale * constraint_gradient
    projected_fraction = projected_norm / constraint_norm.clamp_min(1.0e-12)
    return FusionDiagnostics(
        mode=mode,
        base_norm=float(base_norm.detach().cpu()),
        constraint_norm=float(constraint_norm.detach().cpu()),
        cosine=float(cosine.detach().cpu()),
        projected_constraint_norm=float(projected_norm.detach().cpu()),
        projected_fraction=float(projected_fraction.detach().cpu()),
    )


def _room_tokens(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = probability.mean(dim=(-2, -1))
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(
            0.0,
            1.0,
            probability.shape[-2],
            device=probability.device,
        ),
        torch.linspace(
            0.0,
            1.0,
            probability.shape[-1],
            device=probability.device,
        ),
        indexing="ij",
    )
    denominator = probability.sum(dim=(-2, -1)).clamp_min(1.0e-6)
    center_x = (
        probability * grid_x[None, None]
    ).sum(dim=(-2, -1)) / denominator
    center_y = (
        probability * grid_y[None, None]
    ).sum(dim=(-2, -1)) / denominator
    flattened = probability.flatten(2)
    adjacency = torch.bmm(flattened, flattened.transpose(1, 2))
    adjacency = adjacency / adjacency.amax(dim=(1, 2), keepdim=True).clamp_min(
        1.0e-6
    )
    degree = adjacency.sum(dim=-1)
    tokens = torch.stack(
        [mass, center_x, center_y, degree / max(probability.shape[1] - 1, 1)],
        dim=-1,
    )
    classes = torch.eye(
        probability.shape[1],
        device=probability.device,
    )[None].expand(probability.shape[0], -1, -1)
    return tokens, classes


def room_graph_edit_proxy(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft GED over fixed room channels; never used as a hard GED result."""

    predicted_tokens, predicted_classes = _room_tokens(predicted_probability)
    target_tokens, target_classes = _room_tokens(target_probability.detach())
    predicted_adjacency = torch.bmm(
        predicted_probability.flatten(2),
        predicted_probability.flatten(2).transpose(1, 2),
    )
    target_adjacency = torch.bmm(
        target_probability.flatten(2),
        target_probability.flatten(2).transpose(1, 2),
    )
    predicted_adjacency = predicted_adjacency / predicted_adjacency.amax(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1.0e-6)
    target_adjacency = target_adjacency / target_adjacency.amax(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1.0e-6)
    return soft_graph_edit_distance(
        predicted_tokens,
        target_tokens,
        predicted_classes,
        target_classes,
        predicted_adjacency,
        target_adjacency,
        predicted_valid=predicted_tokens[..., 0] > 1.0e-6,
        target_valid=target_tokens[..., 0] > 1.0e-6,
        aligned_node_order=False,
        aligned_edge_order=False,
    )


__all__ = [
    "FusionDiagnostics",
    "apply_constraint_fusion",
    "room_graph_edit_proxy",
]
