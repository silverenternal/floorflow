"""Audit real canonical-plan tokenization under fixed-point tolerance."""

from __future__ import annotations

from floorflow.data.taxonomy import FloorplanTaxonomy
from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    canonical_fingerprint,
)
from floorflow.representation.floorplan_token_io import CoordinateQuantizer
from floorflow.representation.unified_floorplan_tokens import (
    decode_floorplan,
    tokenize_floorplan,
)


def audit_rplan_token_roundtrip(
    plan: CanonicalFloorplan,
    taxonomy: FloorplanTaxonomy,
    *,
    max_length: int = 512,
    quantization_scale: int = 16,
) -> dict[str, object]:
    """Check structural preservation and finite-token re-encoding."""

    quantizer = CoordinateQuantizer(scale=int(quantization_scale))
    encoded = tokenize_floorplan(plan, taxonomy, quantizer=quantizer)
    decoded = decode_floorplan(encoded.tokens, taxonomy)
    reencoded = tokenize_floorplan(decoded, taxonomy, quantizer=quantizer)
    before = (
        len(plan.rooms),
        len(plan.walls),
        len(plan.openings),
        len(plan.junctions),
        len(plan.adjacency),
    )
    after = (
        len(decoded.rooms),
        len(decoded.walls),
        len(decoded.openings),
        len(decoded.junctions),
        len(decoded.adjacency),
    )
    max_error = float(encoded.max_quantization_error)
    half_step = 0.5 / float(max(1, int(quantization_scale)))
    return {
        "passed": (
            before == after
            and reencoded.text == encoded.text
            and max_error <= half_step + 1.0e-6
            and encoded.token_count <= int(max_length)
        ),
        "structural_counts_match": before == after,
        "token_text_exact": reencoded.text == encoded.text,
        "canonical_fingerprint_exact": (
            canonical_fingerprint(plan)
            == canonical_fingerprint(decoded)
        ),
        "max_quantization_error": max_error,
        "half_step": half_step,
        "token_count": int(encoded.token_count),
        "structural_counts": {
            "rooms": before[0],
            "walls": before[1],
            "openings": before[2],
            "junctions": before[3],
            "adjacency": before[4],
        },
    }


__all__ = ["audit_rplan_token_roundtrip"]
