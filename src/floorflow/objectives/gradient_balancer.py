"""Online gradient-norm balancing with bounded, smooth weight updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GradientNormBalancer:
    base_weights: dict[str, float]
    reference: str = "flow"
    momentum: float = 0.95
    minimum_scale: float = 0.10
    maximum_scale: float = 2.0
    update_interval: int = 50
    exponent: float = 1.0
    norm_ema: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    last_norms: dict[str, float] = field(default_factory=dict)
    last_active_parameters: dict[str, int] = field(default_factory=dict)
    last_update_step: int = -1

    def __post_init__(self) -> None:
        if self.reference not in self.base_weights:
            raise ValueError("reference loss must have a base weight")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0,1)")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive")
        self.scales = {name: 1.0 for name in self.base_weights}
        self.norm_ema = {}

    def _gradient_norm(
        self,
        loss: torch.Tensor,
        parameters: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """Return a norm without allocating one concatenated gradient buffer."""

        norm, _ = self._gradient_summary(loss, parameters)
        return norm

    def _gradient_summary(
        self,
        loss: torch.Tensor,
        parameters: list[torch.nn.Parameter],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gradient norm and the number of touched tensor elements."""

        if not loss.requires_grad:
            zero = torch.zeros((), device=loss.device)
            return zero, zero
        trainable = [
            parameter
            for parameter in parameters
            if parameter.requires_grad
        ]
        if not trainable:
            zero = torch.zeros((), device=loss.device)
            return zero, zero
        gradients = torch.autograd.grad(
            loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        present = [
            gradient.detach()
            for gradient in gradients
            if gradient is not None
        ]
        if not present:
            zero = torch.zeros((), device=loss.device)
            return zero, zero
        if hasattr(torch, "_foreach_norm"):
            squared = torch.stack(
                torch._foreach_norm(present, 2.0)
            ).float().square()
        else:
            squared = torch.stack(
                [gradient.float().square().sum() for gradient in present]
            )
        active = torch.stack(
            [
                torch.count_nonzero(gradient).to(loss.device)
                for gradient in present
            ]
        ).sum()
        return squared.sum().sqrt(), active.to(loss.dtype)

    def maybe_update(
        self,
        losses: dict[str, torch.Tensor],
        parameters: list[torch.nn.Parameter],
        step: int,
    ) -> None:
        if int(step) <= 0 or step % int(self.update_interval) != 0:
            return
        device_summaries = {
            name: self._gradient_summary(loss, parameters)
            for name, loss in losses.items()
            if (
                name in self.base_weights
                and self.base_weights[name] > 0.0
            )
        }
        names = list(device_summaries)
        packed = (
            torch.stack(
                [
                    value
                    for name in names
                    for value in device_summaries[name]
                ]
            )
            .cpu()
            .tolist()
        )
        norms = {
            name: float(packed[index * 2])
            for index, name in enumerate(names)
        }
        active = {
            name: int(packed[index * 2 + 1])
            for index, name in enumerate(names)
        }
        self.last_norms = norms
        self.last_active_parameters = active
        self.last_update_step = int(step)
        for name, norm in norms.items():
            previous = self.norm_ema.get(name, norm)
            self.norm_ema[name] = (
                self.momentum * previous + (1.0 - self.momentum) * norm
            )
        reference_norm = max(self.norm_ema.get(self.reference, 0.0), 1.0e-8)
        for name, norm in self.norm_ema.items():
            if name == self.reference:
                self.scales[name] = 1.0
                continue
            raw = (
                reference_norm / max(norm, 1.0e-8)
            ) ** float(self.exponent)
            self.scales[name] = min(
                self.maximum_scale,
                max(self.minimum_scale, raw),
            )

    def weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [
            self.base_weights[name] * self.scales.get(name, 1.0) * loss
            for name, loss in losses.items()
            if (
                name in self.base_weights
                and self.base_weights[name] > 0.0
            )
        ]
        if not terms:
            raise ValueError("no terms are registered with the gradient balancer")
        return torch.stack(terms).sum()

    def diagnostics(self) -> dict[str, object]:
        return {
            "last_update_step": int(self.last_update_step),
            "reference": self.reference,
            "last_norms": dict(self.last_norms),
            "last_active_parameters": dict(self.last_active_parameters),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "gradient_norm",
            "base_weights": dict(self.base_weights),
            "reference": self.reference,
            "norm_ema": dict(self.norm_ema),
            "scales": dict(self.scales),
            "last_norms": dict(self.last_norms),
            "last_active_parameters": dict(self.last_active_parameters),
            "last_update_step": int(self.last_update_step),
            "exponent": float(self.exponent),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        saved_weights = set(state.get("base_weights", {}))
        if not saved_weights.issubset(self.base_weights):
            raise ValueError("checkpoint has unknown gradient balancer terms")
        if str(state.get("reference", self.reference)) != self.reference:
            # Gradient norms are not comparable across reference objectives.
            # Reusing old scales after a scope/reference migration can silently
            # suppress the newly selected primary metric.
            self.scales = {name: 1.0 for name in self.base_weights}
            self.norm_ema = {}
            self.last_norms = {}
            self.last_active_parameters = {}
            self.last_update_step = -1
            return
        self.norm_ema = {
            name: float(value)
            for name, value in state.get("norm_ema", {}).items()
        }
        self.scales = {
            name: float(state.get("scales", {}).get(name, 1.0))
            for name in self.base_weights
        }
        self.last_norms = {
            name: float(value)
            for name, value in state.get("last_norms", {}).items()
        }
        self.last_active_parameters = {
            name: int(value)
            for name, value in state.get("last_active_parameters", {}).items()
        }
        self.last_update_step = int(state.get("last_update_step", -1))
        self.exponent = float(state.get("exponent", self.exponent))


__all__ = ["GradientNormBalancer"]
