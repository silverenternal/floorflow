"""Mask-ratio curriculum that keeps evaluation anchors represented."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskRatioCurriculum:
    minimum: float = 0.05
    maximum: float = 0.90
    probe_ratio: float = 0.25
    zero_control_probability: float = 0.20
    full_control_probability: float = 0.05
    probe_probability: float = 0.15

    def sample(
        self,
        progress: float,
        *,
        device: torch.device | str = "cpu",
    ) -> float:
        select, shape, jitter = torch.rand(3, device=device)
        if float(select) < self.zero_control_probability:
            return 1.0
        if float(select) < (
            self.zero_control_probability + self.full_control_probability
        ):
            return 0.0
        if float(select) < (
            self.zero_control_probability
            + self.full_control_probability
            + self.probe_probability
        ):
            return self.probe_ratio
        center = 0.15 + 0.35 * min(1.0, max(0.0, float(progress)))
        ratio = center + 0.35 * (float(shape) - 0.5)
        ratio += 0.05 * (float(jitter) - 0.5)
        return min(self.maximum, max(self.minimum, ratio))


__all__ = ["MaskRatioCurriculum"]
