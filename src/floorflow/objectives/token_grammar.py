"""Constrained token grammar helpers for canonical floorplans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from floorflow.representation.unified_floorplan_tokens import FloorplanToken

TOKEN_VOCABULARY_VERSION = "floorflow-token-vocabulary-v1"
TOKEN_STAGES = (
    "envelope",
    "program",
    "rooms",
    "walls",
    "junctions",
    "openings",
    "furniture",
    "validation",
)
STAGE_INDEX = {name: index for index, name in enumerate(TOKEN_STAGES)}


@lru_cache(maxsize=32)
def _vocabulary_index(
    tokens: tuple[str, ...],
) -> dict[str, int]:
    return {value: index for index, value in enumerate(tokens)}


@lru_cache(maxsize=32)
def _vocabulary_kind_names(
    tokens: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(sorted({value.split("|", 1)[0] for value in tokens}))


@lru_cache(maxsize=32)
def _vocabulary_kind_ids(
    tokens: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    output: dict[str, list[int]] = {}
    for token_id, value in enumerate(tokens):
        output.setdefault(value.split("|", 1)[0], []).append(token_id)
    return {kind: tuple(ids) for kind, ids in output.items()}


@lru_cache(maxsize=128)
def _allowed_mask_cpu(
    tokens: tuple[str, ...],
    allowed_kinds: tuple[str, ...],
) -> torch.Tensor:
    allowed = set(allowed_kinds)
    mask = torch.zeros(len(tokens) + 1, dtype=torch.bool)
    for token_id, value in enumerate(tokens):
        mask[token_id] = value.split("|", 1)[0] in allowed
    return mask


@dataclass(frozen=True)
class FloorplanTokenVocabulary:
    tokens: tuple[str, ...]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    version: str = TOKEN_VOCABULARY_VERSION

    def __post_init__(self) -> None:
        if self.pad_token in self.tokens or self.unk_token in self.tokens:
            raise ValueError("special tokens must not be in the content vocabulary")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("token vocabulary must be unique")
        if not self.tokens:
            raise ValueError("token vocabulary must not be empty")

    @property
    def pad_id(self) -> int:
        return len(self.tokens) + 1

    @property
    def unk_id(self) -> int:
        return len(self.tokens)

    @property
    def model_vocab_size(self) -> int:
        return len(self.tokens) + 1

    @property
    def kind_names(self) -> tuple[str, ...]:
        return _vocabulary_kind_names(self.tokens)

    @classmethod
    def from_tokens(
        cls,
        sequences: Iterable[Iterable[FloorplanToken]],
    ) -> FloorplanTokenVocabulary:
        values = sorted(
            {
                token.render()
                for sequence in sequences
                for token in sequence
            }
        )
        return cls(tuple(values))

    def index(self, token: FloorplanToken | str) -> int:
        value = token.render() if isinstance(token, FloorplanToken) else str(token)
        return _vocabulary_index(self.tokens).get(value, self.unk_id)

    def decode(self, token_id: int) -> str:
        value = int(token_id)
        if 0 <= value < len(self.tokens):
            return self.tokens[value]
        if value == self.unk_id:
            return self.unk_token
        if value == self.pad_id:
            return self.pad_token
        raise ValueError(f"token id outside vocabulary: {token_id}")

    def kind(self, token_id: int) -> str:
        return self.decode(token_id).split("|", 1)[0].removeprefix("<").removesuffix(">")

    def encode(
        self,
        tokens: Iterable[FloorplanToken],
        *,
        max_length: int | None = None,
    ) -> list[int]:
        encoded = [self.index(token) for token in tokens]
        if max_length is not None:
            encoded = encoded[: int(max_length)]
        return encoded

    def encode_with_mask(
        self,
        tokens: Iterable[FloorplanToken],
        *,
        max_length: int | None = None,
    ) -> tuple[list[int], list[bool]]:
        values = tuple(tokens)
        if max_length is not None:
            values = values[: int(max_length)]
        ids = [self.index(token) for token in values]
        known = [token.render() in self.tokens for token in values]
        return ids, known

    def sha256(self) -> str:
        payload = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> FloorplanTokenVocabulary:
        version = str(payload.get("version", TOKEN_VOCABULARY_VERSION))
        if version != TOKEN_VOCABULARY_VERSION:
            raise ValueError(f"unsupported token vocabulary version: {version}")
        return cls(
            tuple(str(value) for value in payload["tokens"]),
            pad_token=str(payload.get("pad_token", "<PAD>")),
            unk_token=str(payload.get("unk_token", "<UNK>")),
            version=version,
        )

    @classmethod
    def read(cls, path: str | Path) -> FloorplanTokenVocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = cls.from_json(payload)
        expected = payload.get("sha256")
        if expected is not None and str(expected) != vocabulary.sha256():
            raise ValueError("token vocabulary checksum mismatch")
        return vocabulary

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {**self.to_json(), "sha256": self.sha256()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


_FIXED_PREFIX = (
    "SCHEMA",
    "CASE",
    "TRACK",
    "SIZE",
    "SOURCE_HASH",
)
_BODY_STARTS = {
    "OPERATION",
    "ROOM_BEGIN",
    "ADJACENCY",
    "WALL",
    "JUNCTION",
    "OPENING",
    "FURNITURE_BEGIN",
    "END",
}


def _ring_context(prefix: Sequence[FloorplanToken]) -> str | None:
    for token in reversed(prefix):
        if token.kind.endswith("_BEGIN"):
            return token.kind.removesuffix("_BEGIN")
    return None


_KIND_STAGE = {
    "SCHEMA": "envelope",
    "CASE": "envelope",
    "TRACK": "envelope",
    "SIZE": "envelope",
    "SOURCE_HASH": "envelope",
    "QUANTIZER": "envelope",
    "COORDINATES": "envelope",
    "FOOTPRINT_BEGIN": "envelope",
    "OPERATION": "program",
    "ROOM_BEGIN": "rooms",
    "EXTERIOR_BEGIN": "rooms",
    "HOLE_BEGIN": "rooms",
    "SUPPORT_BEGIN": "rooms",
    "ROOM_END": "rooms",
    "ADJACENCY": "rooms",
    "WALL": "walls",
    "JUNCTION": "junctions",
    "OPENING": "openings",
    "FURNITURE_BEGIN": "furniture",
    "POLYGON_BEGIN": "furniture",
    "FURNITURE_END": "furniture",
    "END": "validation",
}


def stage_for_prefix(prefix: Sequence[FloorplanToken]) -> str:
    """Infer the CAD stage that owns the next token without future labels."""

    for token in reversed(prefix):
        if token.kind in {"VERTEX", "VD", "RING_END"}:
            continue
        return _KIND_STAGE.get(token.kind, "validation")
    return "envelope"


def stage_ids_for_tokens(
    tokens: Sequence[FloorplanToken],
) -> list[int]:
    return [
        STAGE_INDEX[stage_for_prefix(tokens[:index])]
        for index in range(len(tokens))
    ]


def allowed_next_kinds(prefix: Sequence[FloorplanToken]) -> frozenset[str]:
    """Return the grammar-valid token kinds after a prefix."""

    length = len(prefix)
    if length < len(_FIXED_PREFIX):
        return frozenset({_FIXED_PREFIX[length]})
    last = prefix[-1].kind
    if length == len(_FIXED_PREFIX):
        return frozenset({"QUANTIZER", "FOOTPRINT_BEGIN"})
    if last == "QUANTIZER":
        return frozenset({"COORDINATES", "FOOTPRINT_BEGIN"})
    if last == "COORDINATES":
        return frozenset({"FOOTPRINT_BEGIN"})
    if last == "FOOTPRINT_BEGIN":
        return frozenset({"VERTEX"})
    if last == "VERTEX":
        return frozenset({"VERTEX", "VD", "RING_END"})
    if last == "VD":
        return frozenset({"VD", "RING_END"})
    if last == "RING_END":
        context = _ring_context(prefix)
        if context == "FOOTPRINT":
            return frozenset(_BODY_STARTS)
        if context == "EXTERIOR":
            return frozenset({"HOLE_BEGIN", "SUPPORT_BEGIN", "ROOM_END"})
        if context == "HOLE":
            return frozenset({"HOLE_BEGIN", "SUPPORT_BEGIN", "ROOM_END"})
        if context == "SUPPORT":
            return frozenset({"ROOM_END"})
        if context == "POLYGON":
            return frozenset({"FURNITURE_END"})
    if last in {"ROOM_BEGIN"}:
        return frozenset({"EXTERIOR_BEGIN"})
    if last in {
        "EXTERIOR_BEGIN",
        "HOLE_BEGIN",
        "SUPPORT_BEGIN",
        "POLYGON_BEGIN",
    }:
        return frozenset({"VERTEX"})
    if last == "FURNITURE_BEGIN":
        return frozenset({"POLYGON_BEGIN"})
    if last == "ROOM_END":
        return frozenset({"ROOM_BEGIN", *_BODY_STARTS - {"ROOM_BEGIN"}})
    if last == "FURNITURE_END":
        return frozenset({"FURNITURE_BEGIN", "END"})
    if last == "END":
        return frozenset()
    return frozenset(_BODY_STARTS)


def grammar_allowed_mask(
    prefix: Sequence[FloorplanToken],
    vocabulary: FloorplanTokenVocabulary,
) -> torch.Tensor:
    allowed = allowed_next_kinds(prefix)
    mask = torch.zeros(vocabulary.model_vocab_size + 1, dtype=torch.bool)
    for index, value in enumerate(vocabulary.tokens):
        if value.split("|", 1)[0] in allowed:
            mask[index] = True
    return mask


def grammar_prefix_violations(
    tokens: Sequence[FloorplanToken],
) -> list[dict[str, int | str]]:
    violations = []
    for index, token in enumerate(tokens):
        allowed = allowed_next_kinds(tokens[:index])
        if allowed and token.kind not in allowed:
            violations.append(
                {
                    "index": index,
                    "found": token.kind,
                    "allowed": "|".join(sorted(allowed)),
                }
            )
    return violations


def constrained_argmax(
    logits: torch.Tensor,
    prefix: Sequence[FloorplanToken],
    vocabulary: FloorplanTokenVocabulary,
    *,
    allow_end_fallback: bool = True,
) -> int:
    """Select the best grammar-valid token id from one-step logits."""

    if logits.ndim != 1 or logits.numel() != len(vocabulary.tokens):
        raise ValueError("logits must have one value per vocabulary token")
    mask = grammar_allowed_mask(prefix, vocabulary)[: len(vocabulary.tokens)]
    if mask.any():
        return int(logits.masked_fill(~mask, -torch.inf).argmax().item())
    if allow_end_fallback:
        end_ids = [
            index
            for index, value in enumerate(vocabulary.tokens)
            if value.split("|", 1)[0] == "END"
        ]
        if end_ids:
            return int(end_ids[0])
    raise ValueError("no grammar-valid token is available")


class FloorplanGrammarState:
    """Incremental grammar state with cached vocabulary-kind ids."""

    def __init__(
        self,
        vocabulary: FloorplanTokenVocabulary,
        prefix: Sequence[FloorplanToken],
    ) -> None:
        self.vocabulary = vocabulary
        self.prefix = list(prefix)
        self._kind_ids = _vocabulary_kind_ids(vocabulary.tokens)
        self.allowed_kinds = allowed_next_kinds(self.prefix)

    def allowed_mask(self, device: torch.device) -> torch.Tensor:
        return _allowed_mask_cpu(
            self.vocabulary.tokens,
            tuple(sorted(self.allowed_kinds)),
        ).to(device)

    def append(self, token: FloorplanToken) -> None:
        if self.allowed_kinds and token.kind not in self.allowed_kinds:
            raise ValueError(
                f"grammar rejects {token.kind}; expected {sorted(self.allowed_kinds)}"
            )
        self.prefix.append(token)
        self.allowed_kinds = allowed_next_kinds(self.prefix)


def token_sequence_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    grammar_mask: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("token logits/targets must align as [B,L,V]/[B,L]")
    if valid is None:
        valid = (target_ids >= 0) & (target_ids < logits.shape[-1])
    selected_logits = logits[valid]
    selected_targets = target_ids[valid]
    if grammar_mask is not None:
        if grammar_mask.shape != logits.shape:
            raise ValueError("grammar_mask must match token logits")
        selected_logits = selected_logits.masked_fill(
            ~grammar_mask[valid],
            -torch.inf,
        )
    selected_weights = (
        token_weights[valid]
        if token_weights is not None
        else selected_targets.new_ones(selected_targets.shape, dtype=logits.dtype)
    )
    if selected_weights.shape != selected_targets.shape:
        raise ValueError("token_weights must match target ids")
    losses = torch.nn.functional.cross_entropy(
        selected_logits,
        selected_targets,
        reduction="none",
    )
    weighted = losses.mul(selected_weights)
    return weighted.sum() / selected_weights.sum().clamp_min(1.0)


def grammar_masks_for_ids(
    target_ids: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct grammar masks from canonical ids without external labels."""

    if target_ids.ndim != 2:
        raise ValueError("target_ids must be [B,L]")
    output = torch.zeros(
        *target_ids.shape,
        vocabulary.model_vocab_size,
        dtype=torch.bool,
        device=target_ids.device,
    )
    if valid is None:
        valid = torch.ones_like(target_ids, dtype=torch.bool)
    for batch, sequence in enumerate(target_ids.detach().cpu().tolist()):
        prefix: list[FloorplanToken] = []
        for index, token_id in enumerate(sequence):
            output[batch, index] = grammar_allowed_mask(
                prefix,
                vocabulary,
            )[: vocabulary.model_vocab_size].to(output.device)
            if not bool(valid[batch, index]):
                continue
            value = vocabulary.decode(int(token_id))
            if "|" not in value and value.startswith("<"):
                continue
            prefix.append(FloorplanToken.parse(value))
    return output


def token_kind_weights(
    target_ids: torch.Tensor,
    vocabulary: FloorplanTokenVocabulary,
    kind_weights: dict[str, float] | None,
) -> torch.Tensor | None:
    if not kind_weights:
        return None
    weights = torch.ones_like(target_ids, dtype=torch.float32)
    for batch, sequence in enumerate(target_ids.detach().cpu().tolist()):
        for index, token_id in enumerate(sequence):
            if int(token_id) < 0 or int(token_id) >= vocabulary.model_vocab_size:
                continue
            weights[batch, index] = float(
                kind_weights.get(vocabulary.kind(int(token_id)), 1.0)
            )
    return weights.to(target_ids.device)


def stage_weights_for_ids(
    stage_ids: torch.Tensor,
    stage_weights: dict[str, float] | None,
) -> torch.Tensor | None:
    if not stage_weights:
        return None
    weights = torch.tensor(
        [float(stage_weights.get(name, 1.0)) for name in TOKEN_STAGES],
        dtype=torch.float32,
        device=stage_ids.device,
    )
    if stage_ids.numel() and (
        stage_ids.min() < 0 or stage_ids.max() >= len(TOKEN_STAGES)
    ):
        raise ValueError("stage ids must index TOKEN_STAGES")
    return weights[stage_ids.long()]


__all__ = [
    "STAGE_INDEX",
    "TOKEN_STAGES",
    "TOKEN_VOCABULARY_VERSION",
    "FloorplanGrammarState",
    "FloorplanTokenVocabulary",
    "allowed_next_kinds",
    "constrained_argmax",
    "grammar_allowed_mask",
    "grammar_masks_for_ids",
    "grammar_prefix_violations",
    "stage_for_prefix",
    "stage_ids_for_tokens",
    "stage_weights_for_ids",
    "token_kind_weights",
    "token_sequence_loss",
]
