"""Audit the vector-first token grammar without opening an evaluation split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from floorflow.evaluation.canonical_floorplan import (
    CanonicalFloorplan,
    CanonicalOperation,
    canonical_fingerprint,
)
from floorflow.representation.unified_floorplan_tokens import (
    CoordinateQuantizer,
    FloorplanToken,
    decode_floorplan,
    render_floorplan_rasters,
    tokenize_floorplan,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/reports/unified_floorplan_token_audit_v1.json"


def _load_protocol_fixture():
    path = ROOT / "scripts/audit_sota_benchmark_protocol.py"
    spec = importlib.util.spec_from_file_location("unified_token_protocol_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical protocol fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _holed_plan(plan: CanonicalFloorplan) -> CanonicalFloorplan:
    living = plan.rooms[0]
    hole = ((5, 5), (7, 5), (7, 7), (5, 7))
    return replace(
        plan,
        rooms=(replace(living, holes=(hole,)), *plan.rooms[1:]),
        operations=(CanonicalOperation("add_hole", living.id, "not_a_repair"),),
    )


def _order_variant(plan: CanonicalFloorplan) -> CanonicalFloorplan:
    return replace(
        plan,
        footprint=tuple(reversed(plan.footprint)),
        rooms=tuple(
            replace(
                room,
                polygon=tuple(reversed(room.polygon)),
                support=tuple(reversed(room.support or ())),
                holes=tuple(tuple(reversed(hole)) for hole in room.holes),
            )
            for room in reversed(plan.rooms)
        ),
        walls=tuple(
            replace(wall, start=wall.end, end=wall.start)
            for wall in reversed(plan.walls)
        ),
        openings=tuple(reversed(plan.openings)),
        furniture=tuple(reversed(plan.furniture)),
        adjacency=tuple((right, left) for left, right in reversed(plan.adjacency)),
        operations=tuple(reversed(plan.operations)),
    )


def _latencies(operation, repeats: int = 40) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        values.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": max(values),
    }


def _token_mix(tokens: tuple[FloorplanToken, ...]) -> dict[str, int]:
    return dict(sorted(Counter(token.kind for token in tokens).items()))


def _audit_case(name: str, plan: CanonicalFloorplan, taxonomy) -> dict:
    encoded = tokenize_floorplan(plan, taxonomy)
    decoded = decode_floorplan(encoded.tokens, taxonomy)
    reencoded = tokenize_floorplan(decoded, taxonomy)
    variant = tokenize_floorplan(_order_variant(plan), taxonomy)
    rasters = render_floorplan_rasters(decoded, taxonomy)
    canonical_json = json.dumps(
        encoded.plan.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_elements = (
        len(encoded.plan.rooms)
        + len(encoded.plan.walls)
        + len(encoded.plan.openings)
        + len(encoded.plan.furniture)
        + len(encoded.plan.adjacency)
    )
    return {
        "case": name,
        "track": plan.track,
        "token_schema": encoded.tokens[0].args[0],
        "token_count": encoded.token_count,
        "token_bytes": encoded.byte_count,
        "canonical_json_bytes": len(canonical_json),
        "token_to_json_byte_ratio": encoded.byte_count / max(1, len(canonical_json)),
        "tokens_per_vector_element": encoded.token_count / max(1, source_elements),
        "token_mix": _token_mix(encoded.tokens),
        "exact_quantization": encoded.exact_quantization,
        "max_quantization_error": encoded.max_quantization_error,
        "exact_roundtrip": canonical_fingerprint(decoded)
        == canonical_fingerprint(encoded.plan),
        "exact_reencode": reencoded.text == encoded.text,
        "ordering_winding_invariant": variant.text == encoded.text,
        "explicit_end_token": encoded.tokens[-1].kind == "END",
        "room_count": len(decoded.rooms),
        "hole_count": sum(len(room.holes) for room in decoded.rooms),
        "junction_count": len(decoded.junctions),
        "operation_count": len(decoded.operations),
        "raster_room_overlap_pixels": rasters["room_overlap_pixels"],
        "raster_boundary_collision_pixels": rasters[
            "room_boundary_collision_pixels"
        ],
        "raster_outside_footprint_pixels": rasters[
            "outside_footprint_pixels"
        ],
        "encode_latency": _latencies(lambda: tokenize_floorplan(plan, taxonomy)),
        "decode_latency": _latencies(lambda: decode_floorplan(encoded.tokens, taxonomy)),
    }


def _expect_rejection(name: str, operation) -> dict:
    try:
        operation()
    except (TypeError, ValueError) as error:
        return {"attack": name, "rejected": True, "error": str(error)[:240]}
    return {"attack": name, "rejected": False, "error": ""}


def _malformed_attacks(
    plan: CanonicalFloorplan,
    taxonomy,
) -> list[dict]:
    encoded = tokenize_floorplan(plan, taxonomy)

    def unknown_wall():
        tokens = [
            replace(token, args=("opening_1", "door:Door", "missing", *token.args[3:]))
            if token.kind == "OPENING"
            else token
            for token in encoded.tokens
        ]
        decode_floorplan(tokens, taxonomy)

    return [
        _expect_rejection(
            "missing_end",
            lambda: decode_floorplan(encoded.tokens[:-1], taxonomy),
        ),
        _expect_rejection(
            "tokens_after_end",
            lambda: decode_floorplan((*encoded.tokens, FloorplanToken("VERTEX", ("0", "0"))), taxonomy),
        ),
        _expect_rejection("unknown_opening_host", unknown_wall),
        _expect_rejection(
            "duplicate_room_id",
            lambda: tokenize_floorplan(
                replace(
                    plan,
                    rooms=(
                        plan.rooms[0],
                        replace(plan.rooms[1], id=plan.rooms[0].id),
                    ),
                ),
                taxonomy,
            ),
        ),
        _expect_rejection(
            "duplicate_adjacency",
            lambda: tokenize_floorplan(
                replace(plan, adjacency=(plan.adjacency[0], plan.adjacency[0])),
                taxonomy,
            ),
        ),
        _expect_rejection(
            "overlapping_rooms",
            lambda: tokenize_floorplan(
                replace(
                    plan,
                    rooms=(
                        plan.rooms[0],
                        replace(
                            plan.rooms[1],
                            polygon=((14, 2), (29, 2), (29, 17), (14, 17)),
                        ),
                    ),
                ),
                taxonomy,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = _load_protocol_fixture()
    taxonomy = fixture.taxonomy()
    base = fixture.sample_plan()
    holed = _holed_plan(base)
    cases = [
        _audit_case("base", base, taxonomy),
        _audit_case("holed_with_operation", holed, taxonomy),
    ]
    snapping = tokenize_floorplan(
        replace(
            base,
            rooms=(
                replace(
                    base.rooms[0],
                    polygon=((2.03, 2), (15, 2), (15, 17), (2, 17)),
                ),
                base.rooms[1],
            ),
        ),
        taxonomy,
        quantizer=CoordinateQuantizer(scale=16),
    )
    attacks = _malformed_attacks(holed, taxonomy)
    report = {
        "schema": "floorflow-unified-floorplan-token-audit-v1",
        "version": "2026-10-30",
        "scope": "synthetic_contract_audit_no_split_access",
        "test_split_opened": False,
        "gpu_used": False,
        "source_fixture": "scripts/audit_sota_benchmark_protocol.py",
        "vector_first_contract": {
            "canonical_object": "CanonicalFloorplan",
            "raster_role": "derived_diagnostic_only",
            "coordinate_quantization_scale": 16,
            "explicit_end_token": "END",
        },
        "cases": cases,
        "coordinate_snapping": {
            "input_x": 2.03,
            "quantized_x": snapping.plan.rooms[0].polygon[0][0],
            "max_quantization_error": snapping.max_quantization_error,
            "exact_quantization": snapping.exact_quantization,
        },
        "malformed_attacks": attacks,
        "gate": {
            "exact_roundtrip_rate": sum(case["exact_roundtrip"] for case in cases)
            / len(cases),
            "exact_reencode_rate": sum(case["exact_reencode"] for case in cases)
            / len(cases),
            "ordering_invariance_rate": sum(
                case["ordering_winding_invariant"] for case in cases
            )
            / len(cases),
            "explicit_end_rate": sum(case["explicit_end_token"] for case in cases)
            / len(cases),
            "malformed_rejection_rate": sum(item["rejected"] for item in attacks)
            / len(attacks),
            "max_token_count": max(case["token_count"] for case in cases),
            "all_gates_passed": all(
                (
                    all(case["exact_roundtrip"] for case in cases),
                    all(case["exact_reencode"] for case in cases),
                    all(case["ordering_winding_invariant"] for case in cases),
                    all(case["explicit_end_token"] for case in cases),
                    all(item["rejected"] for item in attacks),
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
