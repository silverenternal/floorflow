"""Train-only canonical token targets for residential floorplan generation."""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import CanonicalFloorplan
from floorflow.objectives.token_grammar import (
    allowed_next_kinds,
    TOKEN_STAGES,
    FloorplanTokenVocabulary,
    stage_ids_for_tokens,
)
from floorflow.representation.unified_floorplan_tokens import (
    FloorplanToken,
    tokenize_floorplan,
)

TOKEN_TARGET_PROTOCOL_VERSION = "rplan-canonical-token-target-v2-structure"


@dataclass(frozen=True)
class TokenTargetStatistics:
    cases: int
    minimum_length: int
    median_length: float
    p95_length: int
    maximum_length: int
    recommended_max_length: int
    token_count: int
    kind_counts: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "protocol": TOKEN_TARGET_PROTOCOL_VERSION,
            "cases": self.cases,
            "minimum_length": self.minimum_length,
            "median_length": self.median_length,
            "p95_length": self.p95_length,
            "maximum_length": self.maximum_length,
            "recommended_max_length": self.recommended_max_length,
            "token_count": self.token_count,
            "kind_counts": dict(sorted(self.kind_counts.items())),
        }


@dataclass(frozen=True)
class FloorplanTokenBatch:
    ids: torch.Tensor
    mask: torch.Tensor
    loss_mask: torch.Tensor
    stage_ids: torch.Tensor
    grammar_mask: torch.Tensor
    kind_ids: torch.Tensor
    entity_counts: torch.Tensor
    length: int
    oov_count: int

    def to_item(self) -> dict[str, torch.Tensor | int]:
        return {
            "token_ids": self.ids,
            "token_mask": self.mask,
            "token_loss_mask": self.loss_mask,
            "token_stage_ids": self.stage_ids,
            "token_grammar_mask": self.grammar_mask,
            "token_kind_ids": self.kind_ids,
            "token_entity_counts": self.entity_counts,
            "token_length": torch.tensor(self.length, dtype=torch.long),
            "token_oov_count": torch.tensor(self.oov_count, dtype=torch.long),
        }


def _numeric_bin(value: str, bin_size: int) -> str:
    return str(round(float(value) / float(bin_size)))


def normalize_training_token(
    token: FloorplanToken,
    *,
    numeric_bin_size: int = 8,
    geometry_token_mode: str = "reference",
) -> FloorplanToken | None:
    """Remove non-generative identity while preserving structure and CAD stages."""

    if numeric_bin_size <= 0:
        raise ValueError("numeric_bin_size must be positive")
    if geometry_token_mode not in {"reference", "quantized", "exact"}:
        raise ValueError("unknown geometry_token_mode")
    if token.kind == "CASE":
        return FloorplanToken(token.kind, ("<CASE>",))
    if token.kind == "SOURCE_HASH":
        return FloorplanToken(token.kind, ("<SOURCE>",))
    if token.kind == "OPERATION":
        return FloorplanToken(
            token.kind,
            (token.args[0] if token.args else "<UNKNOWN>", "<TARGET>", "<DETAIL>"),
        )
    numeric_fields = {
        "VERTEX": (0, 1),
        "VD": (0, 1),
        "WALL": (1, 2, 3, 4, 5),
        "JUNCTION": (1, 2),
        "OPENING": (3, 4, 5, 6),
        "ROOM_BEGIN": (2,),
        "FURNITURE_BEGIN": (3,),
    }.get(token.kind, ())
    if numeric_fields:
        args = list(token.args)
        for index in numeric_fields:
            if index < len(args):
                args[index] = (
                    args[index]
                    if geometry_token_mode == "exact"
                    else (
                        "<REF>"
                        if geometry_token_mode == "reference"
                        else _numeric_bin(args[index], numeric_bin_size)
                    )
                )
        return FloorplanToken(token.kind, tuple(args))
    return token


def training_token_sequence(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    delta_coordinates: bool = True,
    numeric_bin_size: int = 8,
    geometry_token_mode: str = "reference",
) -> tuple[FloorplanToken, ...]:
    encoded = tokenize_floorplan(
        plan,
        taxonomy,
        delta_coordinates=delta_coordinates,
    )
    return tuple(
        normalized
        for token in encoded.tokens
        if (
            normalized := normalize_training_token(
                token,
                numeric_bin_size=numeric_bin_size,
                geometry_token_mode=geometry_token_mode,
            )
        )
        is not None
    )


def build_floorplan_token_vocabulary(
    sequences: Iterable[Iterable[FloorplanToken]],
) -> tuple[FloorplanTokenVocabulary, TokenTargetStatistics]:
    materialized = [tuple(sequence) for sequence in sequences]
    if not materialized:
        raise ValueError("at least one token sequence is required")
    lengths = [len(sequence) for sequence in materialized]
    kind_counts: dict[str, int] = {}
    for sequence in materialized:
        for token in sequence:
            kind_counts[token.kind] = kind_counts.get(token.kind, 0) + 1
    p95 = int(np.percentile(lengths, 95, method="higher"))
    maximum = max(lengths)
    recommended = max(64, ((maximum + 63) // 64) * 64)
    statistics_value = TokenTargetStatistics(
        cases=len(materialized),
        minimum_length=min(lengths),
        median_length=float(statistics.median(lengths)),
        p95_length=p95,
        maximum_length=maximum,
        recommended_max_length=recommended,
        token_count=sum(lengths),
        kind_counts=kind_counts,
    )
    return (
        FloorplanTokenVocabulary.from_tokens(materialized),
        statistics_value,
    )


class TokenTargetOverflow(ValueError):
    """Raised instead of silently truncating an invalid token target."""


def token_batch_from_plan(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    vocabulary: FloorplanTokenVocabulary,
    *,
    max_length: int,
    numeric_bin_size: int = 8,
    geometry_token_mode: str = "reference",
) -> FloorplanTokenBatch:
    if max_length <= 0:
        raise ValueError("token max_length must be positive")
    tokens = training_token_sequence(
        plan,
        taxonomy,
        numeric_bin_size=numeric_bin_size,
        geometry_token_mode=geometry_token_mode,
    )
    if len(tokens) > int(max_length):
        raise TokenTargetOverflow(
            f"token sequence has {len(tokens)} tokens but max_length is {max_length}"
        )
    ids, known = vocabulary.encode_with_mask(tokens)
    stages = stage_ids_for_tokens(tokens)
    kind_names = vocabulary.kind_names
    kind_lookup = {kind: index for index, kind in enumerate(kind_names)}
    token_id_kind = [kind_lookup.get(token.kind, 0) for token in tokens]
    grammar_mask = np.zeros(
        (len(tokens), vocabulary.model_vocab_size),
        dtype=bool,
    )
    for index in range(len(tokens)):
        allowed = allowed_next_kinds(tokens[:index])
        for token_id, value in enumerate(vocabulary.tokens):
            grammar_mask[index, token_id] = value.split("|", 1)[0] in allowed
    entity_counts = torch.tensor(
        [
            sum(token.kind == "ROOM_BEGIN" for token in tokens),
            sum(token.kind == "WALL" for token in tokens),
            sum(token.kind == "OPENING" for token in tokens),
        ],
        dtype=torch.float32,
    )
    return FloorplanTokenBatch(
        ids=torch.tensor(ids, dtype=torch.long),
        mask=torch.ones(len(ids), dtype=torch.bool),
        loss_mask=torch.tensor(known, dtype=torch.bool),
        stage_ids=torch.tensor(stages, dtype=torch.long),
        grammar_mask=torch.from_numpy(grammar_mask),
        kind_ids=torch.tensor(token_id_kind, dtype=torch.long),
        entity_counts=entity_counts,
        length=len(ids),
        oov_count=sum(not value for value in known),
    )


def token_stage_lengths(tokens: Iterable[FloorplanToken]) -> dict[str, int]:
    counts = {name: 0 for name in TOKEN_STAGES}
    for stage_id in stage_ids_for_tokens(tuple(tokens)):
        counts[TOKEN_STAGES[stage_id]] += 1
    return counts


__all__ = [
    "TOKEN_TARGET_PROTOCOL_VERSION",
    "FloorplanTokenBatch",
    "TokenTargetOverflow",
    "TokenTargetStatistics",
    "build_floorplan_token_vocabulary",
    "normalize_training_token",
    "token_batch_from_plan",
    "token_stage_lengths",
    "training_token_sequence",
]
