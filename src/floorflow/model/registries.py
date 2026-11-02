"""Validated registries for model families and architecture macros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch.nn as nn

from floorflow.model.specs import ArchSpec


ModelFactory = Callable[[ArchSpec, dict[str, object]], nn.Module]


@dataclass(frozen=True)
class RegistryEntry:
    family: str
    factory: ModelFactory
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not family:
            raise ValueError("family must be non-empty")
        if not callable(self.factory):
            raise TypeError("factory must be callable")
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(str(value).strip() for value in self.required_capabilities if str(value).strip()),
        )


class ModelRegistry:
    """Caller-owned registry with deterministic duplicate rejection."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(
        self,
        family: str,
        *,
        capabilities: tuple[str, ...] = (),
    ):
        def decorator(factory: ModelFactory) -> ModelFactory:
            self.add(family, factory, capabilities=capabilities)
            return factory

        return decorator

    def add(
        self,
        family: str,
        factory: ModelFactory,
        *,
        capabilities: tuple[str, ...] = (),
    ) -> RegistryEntry:
        key = str(family).strip()
        if key in self._entries:
            raise ValueError(f"model family already registered: {key}")
        entry = RegistryEntry(key, factory, capabilities)
        self._entries[key] = entry
        return entry

    def get(self, family: str) -> RegistryEntry:
        key = str(family).strip()
        if key not in self._entries:
            raise KeyError(f"unknown model family: {key}")
        return self._entries[key]

    def build(self, spec: ArchSpec, resolved: Mapping[str, object]) -> nn.Module:
        if not isinstance(spec, ArchSpec):
            raise TypeError("spec must be an ArchSpec")
        entry = self.get(spec.family)
        model = entry.factory(spec, dict(resolved))
        if not isinstance(model, nn.Module):
            raise TypeError(f"factory for {spec.family} did not return nn.Module")
        return model

    def ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "family": entry.family,
                "capabilities": entry.required_capabilities,
            }
            for entry in self._entries.values()
        )


DEFAULT_MODEL_REGISTRY = ModelRegistry()


def build_model(spec: ArchSpec, resolved: Mapping[str, object]) -> nn.Module:
    return DEFAULT_MODEL_REGISTRY.build(spec, resolved)


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "ModelRegistry",
    "RegistryEntry",
    "build_model",
]
