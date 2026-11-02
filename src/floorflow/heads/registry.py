"""Registry for auditable floorplan heads and losses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeadSpec:
    name: str
    loss_name: str
    route_name: str
    default_weight: float
    enabled: bool = True
    diagnostics: tuple[str, ...] = ()


class FloorplanHeadRegistry:
    def __init__(self) -> None:
        self._heads: dict[str, HeadSpec] = {}

    def register(self, spec: HeadSpec) -> None:
        if spec.name in self._heads:
            raise ValueError(f"duplicate head: {spec.name}")
        self._heads[spec.name] = spec

    def enabled(self) -> tuple[HeadSpec, ...]:
        return tuple(spec for spec in self._heads.values() if spec.enabled)

    def weights(self) -> dict[str, float]:
        return {spec.loss_name: spec.default_weight for spec in self.enabled()}

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._heads[name].enabled = enabled


def default_floorplan_heads() -> FloorplanHeadRegistry:
    registry = FloorplanHeadRegistry()
    for name, weight in (
        ("semantic", 1.0),
        ("velocity", 1.0),
        ("wall", 0.10),
        ("opening_contact", 0.10),
        ("furniture_containment", 0.10),
        ("room_boundary", 0.10),
        ("adjacency", 0.05),
        ("design_reward", 0.05),
    ):
        registry.register(HeadSpec(name, f"{name}_loss", name, weight))
    return registry
