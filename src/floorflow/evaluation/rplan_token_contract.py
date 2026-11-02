"""Audit canonical token targets against length and entity contracts."""

from __future__ import annotations

from collections import Counter

from floorflow.data.floorplan_token_targets import training_token_sequence
from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import CanonicalFloorplan
from floorflow.objectives.token_grammar import FloorplanTokenVocabulary


def audit_rplan_token_contract(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    vocabulary: FloorplanTokenVocabulary,
    *,
    max_length: int,
    numeric_bin_size: int = 8,
    geometry_token_mode: str = "reference",
) -> dict[str, object]:
    """Return length, OOV, and entity-count checks for one plan."""

    tokens = training_token_sequence(
        plan,
        taxonomy,
        numeric_bin_size=int(numeric_bin_size),
        geometry_token_mode=geometry_token_mode,
    )
    _ids, known = vocabulary.encode_with_mask(tokens)
    counts = Counter(token.kind for token in tokens)
    expected = (len(plan.rooms), len(plan.walls), len(plan.openings))
    observed = (counts["ROOM_BEGIN"], counts["WALL"], counts["OPENING"])
    unknown = Counter(
        token.render()
        for token, is_known in zip(tokens, known, strict=True)
        if not is_known
    )
    length = len(tokens)
    return {
        "passed": length <= int(max_length) and observed == expected,
        "token_length": length,
        "length_overflow": length > int(max_length),
        "oov_count": sum(unknown.values()),
        "unknown_tokens": dict(sorted(unknown.items())),
        "entity_counts": {
            "rooms": observed[0],
            "walls": observed[1],
            "openings": observed[2],
        },
        "expected_entity_counts": {
            "rooms": expected[0],
            "walls": expected[1],
            "openings": expected[2],
        },
        "entity_count_match": observed == expected,
    }


__all__ = ["audit_rplan_token_contract"]
