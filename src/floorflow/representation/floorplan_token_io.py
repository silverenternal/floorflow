"""Language-level primitives for the reversible floorplan token protocol."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import quote

TOKEN_SCHEMA_VERSION = "floorflow-unified-token-v1"

_KIND = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ATOM_SAFE = "-_.:"


@dataclass(frozen=True)
class CoordinateQuantizer:
    """Fixed-point geometry quantization used before lexical tokenization."""

    scale: int = 16

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("coordinate scale must be positive")

    def quantize(self, value: float) -> int:
        if not math.isfinite(float(value)):
            raise ValueError("cannot quantize a non-finite value")
        return round(float(value) * self.scale)

    def dequantize(self, value: int) -> float:
        return int(value) / float(self.scale)


@dataclass(frozen=True)
class FloorplanToken:
    kind: str
    args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _KIND.fullmatch(self.kind):
            raise ValueError(f"invalid token kind: {self.kind!r}")
        for value in self.args:
            if any(delimiter in value for delimiter in ("|", "\t", "\n", "\r")):
                raise ValueError("token fields cannot contain delimiters or newlines")

    def render(self) -> str:
        return "|".join((self.kind, *self.args))

    @classmethod
    def parse(cls, value: str) -> FloorplanToken:
        parts = value.split("|")
        if not parts or not parts[0]:
            raise ValueError("empty token")
        return cls(parts[0], tuple(parts[1:]))


def atom(value: str) -> str:
    """Percent-encode a free-form token field without losing identity."""

    return quote(str(value), safe=_ATOM_SAFE)


def serialize_tokens(tokens: Iterable[FloorplanToken]) -> str:
    return "\t".join(token.render() for token in tokens)


def parse_tokens(text: str) -> tuple[FloorplanToken, ...]:
    values = text.split("\t")
    if not values:
        raise ValueError("empty token sequence")
    return tuple(FloorplanToken.parse(value) for value in values)


__all__ = [
    "TOKEN_SCHEMA_VERSION",
    "CoordinateQuantizer",
    "FloorplanToken",
    "atom",
    "parse_tokens",
    "serialize_tokens",
]
