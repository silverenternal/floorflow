"""Small, measured runtime helpers for chip-flow training."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _parameter_tier(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith("structured_heads.legalizer_response."):
        return "legalizer"
    if name.startswith("structured_heads."):
        return "structured"
    return "core"


def _is_no_decay(name: str) -> bool:
    segments = name.lower().split(".")
    leaf = segments[-1]
    return (
        leaf in {"bias", "scale", "gate", "type_bias"}
        or any("norm" in segment for segment in segments)
        or any(segment == "coordinate_gate" for segment in segments)
        or leaf.endswith(("_bias", "_scale", "_gate"))
    )


def make_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.98),
    fused: bool = True,
    backbone_lr_scale: float = 1.0,
    structured_head_lr_scale: float = 1.0,
    legalizer_head_lr_scale: float = 1.0,
    structured_head_warmup_steps: int = 0,
    legalizer_head_warmup_steps: int = 0,
    include_frozen: bool = False,
) -> torch.optim.AdamW:
    """Build AdamW with a CUDA-fused kernel when the device supports it."""

    if backbone_lr_scale <= 0.0:
        raise ValueError("backbone_lr_scale must be positive")
    if structured_head_lr_scale <= 0.0 or legalizer_head_lr_scale <= 0.0:
        raise ValueError("head learning-rate scales must be positive")
    if structured_head_warmup_steps < 0 or legalizer_head_warmup_steps < 0:
        raise ValueError("head warmup steps must be non-negative")
    use_fused = bool(fused and torch.cuda.is_available())
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and not include_frozen:
            continue
        key = (_parameter_tier(name), _is_no_decay(name))
        buckets.setdefault(key, []).append(parameter)
    scales = {
        "backbone": float(backbone_lr_scale),
        "core": 1.0,
        "structured": float(structured_head_lr_scale),
        "legalizer": float(legalizer_head_lr_scale),
    }
    warmups = {
        "backbone": 0,
        "core": 0,
        "structured": int(structured_head_warmup_steps),
        "legalizer": int(legalizer_head_warmup_steps),
    }
    groups = []
    for (tier, no_decay), parameters in sorted(buckets.items()):
        groups.append(
            {
                "params": parameters,
                "lr_scale": scales[tier],
                "warmup_steps": warmups[tier],
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "group_name": f"{tier}_{'no_decay' if no_decay else 'decay'}",
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=betas,
        fused=use_fused,
    )


def unfreeze_backbone_at_step(
    model: torch.nn.Module,
    unfreeze_step: int | None,
    step: int,
) -> bool:
    """Enable backbone gradients at the configured staged-unfreeze step."""

    if unfreeze_step is None or int(step) < int(unfreeze_step):
        return False
    changed = False
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.") and not parameter.requires_grad:
            parameter.requires_grad_(True)
            changed = True
    return changed


def set_optimizer_group_lrs(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
) -> dict[str, float]:
    """Apply global cosine LR plus per-head warmup and LR scales."""

    applied = {}
    for group in optimizer.param_groups:
        warmup = int(group.get("warmup_steps", 0))
        warmup_scale = (
            1.0
            if warmup <= 0
            else min(1.0, max(0.0, (int(step) + 1) / float(warmup)))
        )
        value = (
            float(base_lr)
            * float(group.get("lr_scale", 1.0))
            * warmup_scale
        )
        group["lr"] = value
        applied[str(group.get("group_name", len(applied)))] = value
    return applied


@torch.no_grad()
def clip_optimizer_gradients_fail_closed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_norm: float,
    *,
    per_group: bool = True,
    materialize: bool = True,
) -> dict[str, object]:
    """Validate and clip with one device-to-host synchronization."""

    invalid = _nonfinite_gradient_names(model)
    if invalid:
        preview = ", ".join(invalid[:8])
        raise FloatingPointError(
            f"non-finite gradients in {len(invalid)} parameters: {preview}"
        )
    if max_norm < 0.0:
        raise ValueError("gradient clip norm must be non-negative")

    clip_groups = (
        optimizer.param_groups
        if per_group
        else [
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ],
                "group_name": "global",
            }
        ]
    )
    device = next(model.parameters()).device
    specs = []
    for group in clip_groups:
        parameters = [
            parameter
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        pre = (
            torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
            if parameters
            else torch.zeros((), device=device)
        )
        coefficient = torch.minimum(
            torch.ones_like(pre),
            float(max_norm) / pre.clamp_min(torch.finfo(pre.dtype).tiny),
        )
        specs.append(
            (
                str(group.get("group_name", len(specs))),
                pre,
                pre * coefficient,
                coefficient,
            )
        )
    pre_stack = torch.stack([spec[1] for spec in specs])
    post_stack = torch.stack([spec[2] for spec in specs])
    global_pre = pre_stack.square().sum().sqrt()
    global_post = post_stack.square().sum().sqrt()
    if not materialize:
        return {
            "global_pre_clip_norm": global_pre,
            "global_post_clip_norm": global_post,
            "groups": {
                name: {
                    "pre_clip_norm": pre,
                    "post_clip_norm": post,
                    "clip_coefficient": coefficient,
                }
                for name, pre, post, coefficient in specs
            },
        }
    resolved = torch.cat(
        (
            torch.stack((global_pre, global_post)),
            pre_stack,
            post_stack,
            torch.stack([spec[3] for spec in specs]),
        )
    ).detach().cpu()
    values = resolved.tolist()
    group_count = len(specs)
    groups = {}
    for index, (name, _pre, _post, _coefficient) in enumerate(specs):
        groups[name] = {
            "pre_clip_norm": values[2 + index],
            "post_clip_norm": values[2 + group_count + index],
            "clip_coefficient": values[2 + 2 * group_count + index],
        }
    return {
        "global_pre_clip_norm": values[0],
        "global_post_clip_norm": values[1],
        "groups": groups,
    }


@torch.no_grad()
def nonfinite_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    """Return parameter names containing NaN or infinity after an update."""

    named = list(model.named_parameters())
    if not named:
        return ()
    invalid = torch.stack(
        [~torch.isfinite(parameter).all() for _, parameter in named]
    ).cpu()
    return tuple(
        name
        for (name, _parameter), is_invalid in zip(named, invalid, strict=True)
        if bool(is_invalid)
    )


@torch.no_grad()
def _nonfinite_gradient_names(model: torch.nn.Module) -> tuple[str, ...]:
    named = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not named:
        return ()
    invalid = torch.stack(
        [~torch.isfinite(gradient).all() for _, gradient in named]
    ).cpu()
    return tuple(
        name
        for (name, _gradient), is_invalid in zip(named, invalid, strict=True)
        if bool(is_invalid)
    )


def ema_decay_at_step(args: Any, step: int) -> float:
    """Return fixed or warmup-adjusted EMA decay."""

    base = float(args.ema_decay)
    mode = str(getattr(args, "ema_mode", "fixed"))
    if mode == "fixed":
        return base
    if mode == "warmup":
        warmup = int(getattr(args, "ema_warmup_steps", 1000))
        if warmup <= 0:
            return base
        progress = min(1.0, max(0.0, (int(step) + 1.0) / float(warmup)))
        return base * progress
    raise ValueError(f"unknown EMA mode: {mode}")


def initial_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detach a device-local copy of every model state tensor."""

    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def update_ema_state_(
    ema: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    decay: float,
) -> None:
    """Update an EMA state in place with one fused foreach operation."""

    floating: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    for name, value in ema.items():
        source = model_state[name]
        if torch.is_floating_point(value):
            floating.append(value)
            sources.append(source.detach())
        else:
            value.copy_(source)
    if floating:
        torch._foreach_lerp_(floating, sources, 1.0 - float(decay))


def cuda_rng_state(device: str | torch.device) -> torch.Tensor | None:
    """Capture the current CUDA generator state for the training device."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    return torch.cuda.get_rng_state(resolved)


def restore_cuda_rng_state(
    state: torch.Tensor | None,
    device: str | torch.device,
) -> None:
    """Restore a saved CUDA generator state when the current device is CUDA."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and state is not None:
        torch.cuda.set_rng_state(state.cpu(), resolved)


def save_torch_atomic(path: str | Path, payload: Any) -> None:
    """Write a torch payload through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write JSON through an atomic rename so interrupted runs stay resumable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _grouped_vector(
    values: list[tuple[str, torch.Tensor]],
    *,
    limit: int | None = None,
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, value in values:
        flat = value.detach().reshape(-1)
        grouped.setdefault(name.split(".", 1)[0], []).append(
            flat[:limit] if limit is not None else flat
        )
    return {
        name: torch.cat(parts)
        for name, parts in grouped.items()
        if parts
    }


@torch.no_grad()
def gradient_norm_by_prefix(model: torch.nn.Module) -> dict[str, float]:
    gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    return {
        name: float(vector.norm())
        for name, vector in _grouped_vector(gradients).items()
    }


@torch.no_grad()
def parameter_probe_by_prefix(
    model: torch.nn.Module,
    *,
    limit: int = 256,
) -> dict[str, torch.Tensor]:
    return _grouped_vector(list(model.named_parameters()), limit=limit)


@torch.no_grad()
def parameter_delta_by_prefix(
    model: torch.nn.Module,
    probe: Mapping[str, torch.Tensor],
    *,
    limit: int = 256,
) -> dict[str, float]:
    current = parameter_probe_by_prefix(model, limit=limit)
    return {
        name: float((current[name] - value).norm())
        for name, value in probe.items()
    }


@torch.no_grad()
def ema_gap_by_prefix(
    model: torch.nn.Module,
    ema: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    values = [
        (name, parameter - ema[name])
        for name, parameter in model.named_parameters()
        if name in ema
    ]
    return {
        name: float(vector.norm())
        for name, vector in _grouped_vector(values).items()
    }


__all__ = [
    "clip_optimizer_gradients_fail_closed",
    "cuda_rng_state",
    "ema_decay_at_step",
    "ema_gap_by_prefix",
    "gradient_norm_by_prefix",
    "initial_ema_state",
    "make_optimizer",
    "nonfinite_parameter_names",
    "parameter_delta_by_prefix",
    "parameter_probe_by_prefix",
    "restore_cuda_rng_state",
    "save_torch_atomic",
    "set_optimizer_group_lrs",
    "unfreeze_backbone_at_step",
    "update_ema_state_",
    "write_json_atomic",
]
