"""Declarative guarded repair strategies for floorplans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from floorflow.inference.room_repair import (
    apply_floorplan_repairs,
    close_room_partition,
    drop_outside_furniture,
    drop_unsupported_openings,
    merge_room_fragments,
)
from floorflow.data.taxonomy import FloorplanTaxonomy


@dataclass(frozen=True)
class RepairStrategy:
    name: str
    failure_family: str
    action: Callable[[torch.Tensor], torch.Tensor]
    guard: Callable[[torch.Tensor, torch.Tensor], bool]


class RepairRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, RepairStrategy] = {}

    def register(self, strategy: RepairStrategy) -> None:
        if strategy.name in self._strategies:
            raise ValueError(f"duplicate repair strategy: {strategy.name}")
        self._strategies[strategy.name] = strategy

    def names(self) -> tuple[str, ...]:
        return tuple(self._strategies)

    def candidates(self, labels: torch.Tensor) -> list[tuple[str, torch.Tensor, bool]]:
        rows = []
        for strategy in self._strategies.values():
            candidate = strategy.action(labels)
            rows.append((strategy.name, candidate, strategy.guard(labels, candidate)))
        return rows


class RepairAcceptanceController:
    def __init__(
        self,
        feature_dim: int,
        centers: list[float] | None = None,
        scales: list[float] | None = None,
    ) -> None:
        self.weights = [0.0] * feature_dim
        self.threshold = 0.0
        self.centers = centers or [0.0] * feature_dim
        self.scales = scales or [1.0] * feature_dim

    def score(self, features: list[float]) -> float:
        normalized = [
            (value - center) / max(scale, 1.0e-8)
            for value, center, scale in zip(features, self.centers, self.scales)
        ]
        return sum(weight * value for weight, value in zip(self.weights, normalized))

    def accept(self, features: list[float]) -> bool:
        return self.score(features) >= self.threshold

    def fit(self, features: list[list[float]], labels: list[bool], epochs: int = 200, lr: float = 0.05) -> None:
        for _ in range(epochs):
            for row, label in zip(features, labels):
                probability = 1.0 / (1.0 + torch.exp(-torch.tensor(self.score(row))).item())
                error = probability - float(label)
                self.weights = [weight - lr * error * value for weight, value in zip(self.weights, row)]


def build_floorplan_repair_registry(
    footprint: torch.Tensor,
    taxonomy: FloorplanTaxonomy,
) -> RepairRegistry:
    registry = RepairRegistry()
    footprint_np = footprint.detach().cpu().numpy().astype("uint8")

    def as_numpy(labels: torch.Tensor):
        return labels.detach().cpu().numpy()

    def as_tensor(values, device):
        return torch.from_numpy(values).to(device=device)

    registry.register(
        RepairStrategy(
            "merge_rooms",
            "room_fragmentation",
            lambda labels: as_tensor(merge_room_fragments(as_numpy(labels), taxonomy), labels.device),
            lambda left, right: right.shape == left.shape,
        )
    )
    registry.register(
        RepairStrategy(
            "drop_unsupported_openings",
            "opening_detached_from_wall",
            lambda labels: as_tensor(drop_unsupported_openings(as_numpy(labels), taxonomy), labels.device),
            lambda left, right: right.shape == left.shape,
        )
    )
    registry.register(
        RepairStrategy(
            "drop_outside_furniture",
            "furniture_outside_room",
            lambda labels: as_tensor(drop_outside_furniture(as_numpy(labels), taxonomy), labels.device),
            lambda left, right: right.shape == left.shape,
        )
    )
    registry.register(
        RepairStrategy(
            "close_rooms",
            "room_partition_gap",
            lambda labels: as_tensor(close_room_partition(as_numpy(labels), footprint_np, taxonomy), labels.device),
            lambda left, right: right.shape == left.shape,
        )
    )
    registry.register(
        RepairStrategy(
            "all_guarded_repairs",
            "combined_validity",
            lambda labels: as_tensor(apply_floorplan_repairs(as_numpy(labels), footprint_np, taxonomy), labels.device),
            lambda left, right: right.shape == left.shape,
        )
    )
    return registry
