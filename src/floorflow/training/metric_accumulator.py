"""Device-resident metric accumulation for training epochs."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class MetricAccumulator:
    """Accumulate scalar metrics without synchronizing at every batch."""

    def __init__(self) -> None:
        self._tensor_totals: dict[str, torch.Tensor] = {}
        self._float_totals: dict[str, float] = {}

    def add(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        weight: float = 1.0,
    ) -> None:
        if torch.is_tensor(value):
            scalar = value.detach()
            if scalar.ndim:
                scalar = scalar.float().mean()
            scalar = scalar.float() * float(weight)
            current = self._tensor_totals.get(name)
            self._tensor_totals[name] = (
                scalar if current is None else current + scalar.to(current.device)
            )
            return
        self._float_totals[name] = (
            self._float_totals.get(name, 0.0) + float(value) * float(weight)
        )

    def finalize(self, denominator: int) -> dict[str, float]:
        scale = 1.0 / max(1, int(denominator))
        names = list(self._tensor_totals)
        output: dict[str, float] = {}
        if names:
            packed = torch.stack(
                [self._tensor_totals[name].detach() for name in names]
            )
            output = {
                name: float(value) * scale
                for name, value in zip(names, packed.cpu().tolist())
            }
        for name, value in self._float_totals.items():
            output[name] = output.get(name, 0.0) + value * scale
        return output


class PackedMetricAccumulator:
    """Accumulate a changing metric map with one vector add per batch."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._indices: dict[str, int] = {}
        self._totals: torch.Tensor | None = None

    def _grow(self, names: list[str], device: torch.device) -> None:
        if self._totals is None:
            self._totals = torch.zeros(len(names), device=device)
        elif len(names) > len(self._names):
            extra = torch.zeros(len(names) - len(self._names), device=device)
            self._totals = torch.cat((self._totals, extra))
        self._names = names
        self._indices = {name: index for index, name in enumerate(names)}

    def add(
        self,
        values: Mapping[str, torch.Tensor | float],
        *,
        weight: float = 1.0,
    ) -> None:
        names = list(self._names)
        names.extend(name for name in values if name not in self._indices)
        tensor_values = [
            value for value in values.values() if torch.is_tensor(value)
        ]
        device = (
            self._totals.device
            if self._totals is not None
            else (
                tensor_values[0].device
                if tensor_values
                else torch.device("cpu")
            )
        )
        self._grow(names, device)
        scalars = [
            (
                value.detach().float()
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=device)
            )
            for value in values.values()
        ]
        packed = torch.stack(
            [value.mean() if value.ndim else value for value in scalars]
        )
        if tuple(values) == tuple(names):
            self._totals.add_(packed, alpha=float(weight))
            return
        expanded = torch.zeros(len(names), device=device)
        for row, name in enumerate(values):
            expanded[self._indices[name]] = packed[row]
        self._totals.add_(expanded, alpha=float(weight))

    def finalize(self, denominator: int) -> dict[str, float]:
        if self._totals is None:
            return {}
        scale = 1.0 / max(1, int(denominator))
        return {
            name: float(value) * scale
            for name, value in zip(
                self._names,
                self._totals.detach().cpu().tolist(),
            )
        }


__all__ = ["MetricAccumulator", "PackedMetricAccumulator"]
