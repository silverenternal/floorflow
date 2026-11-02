"""Validated policy shared by official evaluation and external legalization."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LegalizerPolicy:
    relation: str = "restored"
    reshape: bool = True
    trust: float = 1.3
    outline: bool = True
    time_limit_ms: int = 2000

    def validate(self) -> None:
        if self.relation not in {"restored", "expanded"}:
            raise ValueError("legalizer relation must be restored or expanded")
        if not math.isfinite(float(self.trust)) or self.trust <= 0.0:
            raise ValueError("legalizer trust must be finite and positive")
        if self.time_limit_ms <= 0:
            raise ValueError("legalizer time limit must be positive")

    def kwargs(self) -> dict[str, bool | float | str]:
        self.validate()
        return {
            "relation": self.relation,
            "reshape": self.reshape,
            "trust": float(self.trust),
            "outline": self.outline,
        }


__all__ = ["LegalizerPolicy"]
