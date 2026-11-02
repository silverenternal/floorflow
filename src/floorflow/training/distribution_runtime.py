"""Small runtime helpers shared by distribution training entry points."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from floorflow.flow.cfm_backend import flow_backend_version


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distribution_query_counts(queries: int) -> dict[str, int]:
    return {
        "wall": queries,
        "opening": queries,
        "room": queries,
        "furniture": queries,
        "support": max(1, queries // 2),
    }


def configure_flow_args(args, preset) -> str | None:
    if args.flow_matcher is None:
        args.flow_matcher = preset.matcher
    if args.time_sampling is None:
        args.time_sampling = preset.time_sampling
    if args.flow_sigma is None:
        args.flow_sigma = preset.sigma
    if args.time_sampling_mean is None:
        args.time_sampling_mean = preset.time_sampling_mean
    if args.time_sampling_std is None:
        args.time_sampling_std = preset.time_sampling_std
    backend_version = flow_backend_version(args.flow_matcher)
    if args.flow_matcher != "legacy" and backend_version is None:
        raise SystemExit(
            "TorchCFM is required for the selected matcher; "
            "run `uv sync --extra train`"
        )
    return backend_version


def load_style_assignments(path: str | Path) -> tuple[dict[str, dict[str, int]], int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments = payload.get("assignments", {})
    num_styles = int(payload.get("num_styles", 0))
    if num_styles <= 0 or "train" not in assignments:
        raise ValueError("style manifest requires train assignments")
    return assignments, num_styles


def resolve_style_assignments(args) -> dict[str, dict[str, int]]:
    assignments, num_styles = (
        load_style_assignments(args.style_manifest)
        if args.style_manifest is not None
        else ({}, 0)
    )
    if num_styles:
        args.num_styles = num_styles
    return assignments


def style_ids_for_names(
    names: tuple[str, ...] | list[str],
    assignments: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    try:
        values = [int(assignments[str(name)]) for name in names]
    except KeyError as error:
        raise KeyError(f"style assignment missing for {error.args[0]}") from error
    return torch.tensor(values, device=device, dtype=torch.long)


def apply_style_dropout(
    style_ids: torch.Tensor | None,
    probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    """Randomly omit the optional style prompt while preserving batch shape."""

    if style_ids is None:
        return None
    if not 0.0 <= float(probability) < 1.0:
        raise ValueError("style dropout probability must be in [0, 1)")
    if probability <= 0.0:
        return style_ids
    keep = torch.rand(
        (),
        device=style_ids.device,
        generator=generator,
    ) >= float(probability)
    return style_ids if bool(keep) else None


__all__ = [
    "apply_style_dropout",
    "configure_flow_args",
    "distribution_query_counts",
    "load_style_assignments",
    "resolve_style_assignments",
    "set_training_seed",
    "style_ids_for_names",
]
