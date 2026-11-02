"""Reusable train-only retrieval teacher utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def name_to_cache_path(name: str) -> Path:
    parent, floor = name.split("/", 1)
    floor = floor.replace("__", "/").replace("-", "_")
    return Path(parent) / f"{floor}.npz"


def excluded_indices(
    names: list[str],
    path_to_index: dict[Path, int],
) -> torch.Tensor:
    values = []
    for name in names:
        relative = name_to_cache_path(name)
        if relative not in path_to_index:
            raise KeyError(f"training sample missing from retrieval bank: {relative}")
        values.append(path_to_index[relative])
    return torch.tensor(values, dtype=torch.long)


def distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scaled_student = F.interpolate(
        student,
        size=teacher.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ) / float(temperature)
    scaled_teacher = teacher / float(temperature)
    return F.kl_div(
        scaled_student.log_softmax(dim=1),
        scaled_teacher.softmax(dim=1),
        reduction="batchmean",
    ) * float(temperature**2)


__all__ = ["distillation_loss", "excluded_indices", "name_to_cache_path"]
