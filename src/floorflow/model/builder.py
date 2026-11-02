"""Budget-aware model macro resolution and construction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from floorflow.model.budget import (
    BudgetExceededError,
    count_parameters,
    estimate_output_head_params,
    parameter_table,
    profile_forward_macs,
    select_output_kernels,
)
from floorflow.model.registries import DEFAULT_MODEL_REGISTRY, ModelRegistry
from floorflow.model.specs import ArchSpec, BudgetSpec


RESERVED_BACKBONE_KEYS = {
    "base_channels",
    "dynamic_output_head",
    "input_resolution",
    "levels",
    "output_kernels",
    "target_channels",
}


@dataclass(frozen=True)
class ResolvedArch:
    base_channels: int
    levels: int
    output_kernels: tuple[int, ...]
    estimated_params: int | None = None
    estimated_macs: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "base_channels": self.base_channels,
            "levels": self.levels,
            "output_kernels": list(self.output_kernels),
            "estimated_params": self.estimated_params,
            "estimated_macs": self.estimated_macs,
        }


@dataclass(frozen=True)
class BuiltModel:
    model: nn.Module
    spec: ArchSpec
    resolved: ResolvedArch
    structure_signature: str
    parameter_table: dict[str, int]


def _default_base_channels(spec: ArchSpec) -> int:
    if spec.base_channels is not None:
        return spec.base_channels
    return min(64, spec.budget.max_base_channels)


def _default_levels(spec: ArchSpec, base_channels: int) -> int:
    if spec.levels is not None:
        return spec.levels
    requested = int(math.ceil(math.log2(max(64, min(spec.resolution)) / 64.0))) + 1
    requested = max(spec.budget.min_levels, requested)
    if base_channels >= 96:
        requested -= 1
    return min(spec.budget.max_levels, requested)


def _head_budget(spec: ArchSpec) -> int | None:
    if spec.budget.max_params is None:
        return None
    return max(1, int(spec.budget.max_params * spec.budget.head_fraction))


def _resolve_unchecked(
    spec: ArchSpec,
    base_channels: int,
    levels: int,
    registry: ModelRegistry,
) -> tuple[nn.Module, ResolvedArch]:
    kernels = spec.output_kernels
    if kernels is None:
        kernels = select_output_kernels(
            spec.resolution,
            spec.target_feature_px,
            base_channels,
            spec.target_channels,
            max_params=_head_budget(spec),
        )
    head_params = estimate_output_head_params(base_channels, spec.target_channels, kernels)
    candidate = ResolvedArch(base_channels, levels, kernels, estimated_params=head_params)
    model = registry.build(spec, candidate.as_dict())
    return model, ResolvedArch(
        base_channels,
        levels,
        kernels,
        estimated_params=count_parameters(model),
    )


def _candidate_widths(spec: ArchSpec, start: int) -> list[int]:
    budget = spec.budget
    lower = min(budget.min_base_channels, start)
    width_step = max(1, budget.width_step)
    values: list[int] = []
    value = start
    while value >= lower:
        values.append(value)
        value -= width_step
    if values[-1] != lower:
        values.append(lower)
    return values


def _profile_macs(model: nn.Module, spec: ArchSpec) -> int:
    height, width = spec.resolution
    x_t = torch.zeros(1, spec.target_channels, height, width)
    time = torch.zeros(1)
    footprint = torch.ones(1, 1, height, width)
    with torch.no_grad():
        profile = profile_forward_macs(model, lambda: model(x_t, time, footprint))
    return profile.macs


def structure_signature(model: nn.Module) -> str:
    rows = []
    for name, module in model.named_modules():
        shapes = tuple(tuple(parameter.shape) for parameter in module.parameters(recurse=False))
        rows.append((name, type(module).__name__, shapes))
    payload = repr(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _floorflow_factory(spec: ArchSpec, resolved: Mapping[str, object]) -> nn.Module:
    from floorflow.model.backbone import FloorplanFlowModel

    kwargs = dict(spec.backbone_kwargs)
    conflicts = RESERVED_BACKBONE_KEYS.intersection(kwargs)
    if conflicts:
        raise ValueError(f"backbone_kwargs contain reserved keys: {sorted(conflicts)}")
    return FloorplanFlowModel(
        target_channels=spec.target_channels,
        base_channels=int(resolved["base_channels"]),
        levels=int(resolved["levels"]),
        output_kernels=tuple(int(value) for value in resolved["output_kernels"]),
        dynamic_output_head=spec.dynamic_output,
        input_resolution=spec.resolution,
        **kwargs,
    )


if "floorflow" not in DEFAULT_MODEL_REGISTRY.ids():
    DEFAULT_MODEL_REGISTRY.add("floorflow", _floorflow_factory)


class ModelMacro:
    """Resolve, validate, and construct a model family from an ArchSpec."""

    def __init__(self, spec: ArchSpec, registry: ModelRegistry | None = None) -> None:
        self.spec = spec
        self.registry = registry or DEFAULT_MODEL_REGISTRY

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        registry: ModelRegistry | None = None,
    ) -> "ModelMacro":
        return cls(ArchSpec.from_mapping(value), registry)

    @classmethod
    def from_json(cls, path: str | Path, registry: ModelRegistry | None = None) -> "ModelMacro":
        return cls(ArchSpec.from_json(path), registry)

    @classmethod
    def from_yaml(cls, path: str | Path, registry: ModelRegistry | None = None) -> "ModelMacro":
        return cls(ArchSpec.from_yaml(path), registry)

    def resolve(self) -> ResolvedArch:
        budget = self.spec.budget
        start_width = _default_base_channels(self.spec)
        start_levels = _default_levels(self.spec, start_width)
        if not budget.min_levels <= start_levels <= budget.max_levels:
            raise ValueError("requested levels fall outside the budget range")
        last_error: Exception | None = None
        width_candidates = _candidate_widths(self.spec, start_width)
        level_candidates = list(range(start_levels, budget.min_levels - 1, -1))
        for levels in level_candidates:
            for width in width_candidates:
                try:
                    model, resolved = _resolve_unchecked(
                        self.spec, width, levels, self.registry
                    )
                    if budget.max_params is not None and count_parameters(model) > budget.max_params:
                        last_error = BudgetExceededError(
                            f"parameters {count_parameters(model)} exceed {budget.max_params}"
                        )
                        continue
                    macs = _profile_macs(model, self.spec) if budget.max_macs is not None else None
                    if macs is not None and macs > budget.max_macs:
                        last_error = BudgetExceededError(
                            f"MACs {macs} exceed {budget.max_macs}"
                        )
                        continue
                    return ResolvedArch(
                        resolved.base_channels,
                        resolved.levels,
                        resolved.output_kernels,
                        resolved.estimated_params,
                        macs,
                    )
                except BudgetExceededError as exc:
                    last_error = exc
        raise BudgetExceededError(
            f"no architecture fits {budget.to_dict()}: {last_error}"
        )

    def build_bundle(self) -> BuiltModel:
        resolved = self.resolve()
        previous_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.spec.seed)
            model = self.registry.build(self.spec, resolved.as_dict())
        finally:
            torch.random.set_rng_state(previous_state)
        return BuiltModel(
            model=model,
            spec=self.spec,
            resolved=resolved,
            structure_signature=structure_signature(model),
            parameter_table=parameter_table(model),
        )

    def build(self) -> nn.Module:
        return self.build_bundle().model


def map_legacy_state_dict(
    state: Mapping[str, torch.Tensor],
    accepted_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mapped = dict(state)
    renamed: dict[str, str] = {}
    for prefix in ("module.", "model."):
        for key in list(mapped):
            if key in accepted_keys:
                continue
            stripped = key[len(prefix):] if key.startswith(prefix) else None
            if stripped and stripped in accepted_keys and stripped not in mapped:
                mapped[stripped] = mapped.pop(key)
                renamed[key] = stripped
    return mapped, renamed


def load_state_into_model(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    accepted = set(model.state_dict())
    mapped, renamed = map_legacy_state_dict(state, accepted)
    result = model.load_state_dict(mapped, strict=False)
    return {
        "renamed": renamed,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "loaded_fraction": 1.0 - len(result.missing_keys) / max(1, len(accepted)),
    }


__all__ = [
    "BuiltModel",
    "ModelMacro",
    "ResolvedArch",
    "load_state_into_model",
    "map_legacy_state_dict",
    "structure_signature",
]
