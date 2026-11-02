"""Declarative architecture and resource-budget contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _odd_kernel(value: Any) -> int:
    parsed = _positive_int(value, "kernel size")
    if parsed % 2 == 0:
        raise ValueError(f"kernel sizes must be odd, got {parsed}")
    return parsed


@dataclass(frozen=True)
class BudgetSpec:
    """Hard resource limits used while resolving a model macro."""

    max_params: int | None = None
    max_macs: int | None = None
    min_base_channels: int = 16
    max_base_channels: int = 256
    width_step: int = 8
    min_levels: int = 2
    max_levels: int = 5
    head_fraction: float = 0.15

    def __post_init__(self) -> None:
        for name in ("max_params", "max_macs"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")
            if value is not None:
                object.__setattr__(self, name, int(value))
        for name in ("min_base_channels", "max_base_channels", "width_step", "min_levels", "max_levels"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.min_base_channels > self.max_base_channels:
            raise ValueError("min_base_channels exceeds max_base_channels")
        if self.min_levels > self.max_levels:
            raise ValueError("min_levels exceeds max_levels")
        if not 0.0 < float(self.head_fraction) <= 1.0:
            raise ValueError("head_fraction must be in (0, 1]")
        object.__setattr__(self, "head_fraction", float(self.head_fraction))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BudgetSpec":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_params": self.max_params,
            "max_macs": self.max_macs,
            "min_base_channels": self.min_base_channels,
            "max_base_channels": self.max_base_channels,
            "width_step": self.width_step,
            "min_levels": self.min_levels,
            "max_levels": self.max_levels,
            "head_fraction": self.head_fraction,
        }


@dataclass(frozen=True)
class ArchSpec:
    """Serializable description of a model family and its search envelope."""

    name: str
    family: str = "floorflow"
    target_channels: int = 12
    base_channels: int | None = None
    levels: int | None = None
    resolution: tuple[int, int] = (256, 256)
    target_feature_px: int = 2
    output_kernels: tuple[int, ...] | None = None
    dynamic_output: bool = True
    seed: int = 0
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    backbone_kwargs: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        family = str(self.family).strip()
        if not name:
            raise ValueError("name must be non-empty")
        if not family:
            raise ValueError("family must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "target_channels", _positive_int(self.target_channels, "target_channels"))
        if self.base_channels is not None:
            object.__setattr__(self, "base_channels", _positive_int(self.base_channels, "base_channels"))
        if self.levels is not None:
            object.__setattr__(self, "levels", _positive_int(self.levels, "levels"))
        if len(self.resolution) != 2:
            raise ValueError("resolution must contain height and width")
        resolution = tuple(_positive_int(value, "resolution") for value in self.resolution)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "target_feature_px", _positive_int(self.target_feature_px, "target_feature_px"))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "schema_version", _positive_int(self.schema_version, "schema_version"))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.output_kernels is not None:
            kernels = tuple(sorted({_odd_kernel(value) for value in self.output_kernels}))
            if not kernels:
                raise ValueError("output_kernels cannot be empty")
            object.__setattr__(self, "output_kernels", kernels)
        if not isinstance(self.budget, BudgetSpec):
            object.__setattr__(self, "budget", BudgetSpec.from_mapping(self.budget))
        if not isinstance(self.backbone_kwargs, Mapping):
            raise TypeError("backbone_kwargs must be a mapping")
        object.__setattr__(self, "backbone_kwargs", dict(self.backbone_kwargs))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchSpec":
        payload = dict(value)
        if "budget" in payload:
            payload["budget"] = BudgetSpec.from_mapping(payload["budget"])
        if "resolution" in payload:
            payload["resolution"] = tuple(payload["resolution"])
        if payload.get("output_kernels") is not None:
            payload["output_kernels"] = tuple(payload["output_kernels"])
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ArchSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArchSpec":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML architecture specs") from exc
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "target_channels": self.target_channels,
            "base_channels": self.base_channels,
            "levels": self.levels,
            "resolution": list(self.resolution),
            "target_feature_px": self.target_feature_px,
            "output_kernels": list(self.output_kernels) if self.output_kernels else None,
            "dynamic_output": self.dynamic_output,
            "seed": self.seed,
            "budget": self.budget.to_dict(),
            "backbone_kwargs": dict(self.backbone_kwargs),
            "schema_version": self.schema_version,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["ArchSpec", "BudgetSpec", "SCHEMA_VERSION"]
