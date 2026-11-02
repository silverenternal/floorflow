"""Reproducible CFM training presets for fast and multi-stage curves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowPreset:
    name: str
    matcher: str
    time_sampling: str
    sigma: float
    validation_steps: int
    room_transport_weight: float
    time_sampling_mean: float = 0.0
    time_sampling_std: float = 1.0


PRESETS = {
    "baseline": FlowPreset("baseline", "torchcfm", "uniform", 0.0, 0, 0.0),
    "fast_cfm": FlowPreset(
        "fast_cfm",
        "torchcfm",
        "logit_normal",
        0.05,
        12,
        0.0,
    ),
    "fast_vp": FlowPreset("fast_vp", "torchcfm_vp", "logit_normal", 0.05, 12, 0.0),
    "multi_stage_ot": FlowPreset(
        "multi_stage_ot",
        "torchcfm",
        "logit_normal",
        0.0,
        24,
        0.25,
    ),
}


def resolve_flow_preset(name: str) -> FlowPreset:
    try:
        return PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown flow preset: {name}") from exc


__all__ = ["PRESETS", "FlowPreset", "resolve_flow_preset"]
