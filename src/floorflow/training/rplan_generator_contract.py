"""Canonical decode and validation contract for RPLAN generator training."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from floorflow.evaluation.canonical_floorplan import canonical_fingerprint
from floorflow.evaluation.fevp_metric_contracts import (
    FEVPTolerances,
    watertight_diagnostics,
)
from floorflow.model.unified_floorplan_generator import (
    UnifiedFloorplanGenerator,
    decode_generated_floorplan,
)
from floorflow.representation.unified_floorplan_tokens import (
    decode_floorplan,
    tokenize_floorplan,
    validate_vector_floorplan,
)
from floorflow.representation.vector_floorplan_raster import (
    render_floorplan_rasters,
)
from floorflow.training.floorplan_token_training_runtime import (
    FloorplanTokenTrainingRuntime,
)


def contract_sample(
    model: UnifiedFloorplanGenerator,
    batch: dict,
    taxonomy,
    tolerances: FEVPTolerances,
    token_runtime: FloorplanTokenTrainingRuntime,
) -> dict[str, object]:
    with torch.no_grad():
        output = model(
            batch["boundary_mask"],
            batch["boundary_valid"],
            batch["boundary_points"],
            batch["entrance_points"],
            batch["entrance_valid"],
            **token_runtime.forward_kwargs(batch, training=False),
        )
        return contract_from_output(output, batch, taxonomy, tolerances)


def contract_from_output(
    output: dict[str, torch.Tensor],
    batch: dict,
    taxonomy,
    tolerances: FEVPTolerances,
) -> dict[str, object]:
    try:
        plan = decode_generated_floorplan(
            output,
            batch["boundary_points"],
            taxonomy,
            case_id=batch["case_name"][0],
            entrance_points=batch["entrance_points"],
        )
    except ValueError as error:
        return {
            "case_name": batch["case_name"][0],
            "contract_valid": False,
            "error": str(error),
        }
    try:
        tokenized = tokenize_floorplan(plan, taxonomy)
        decoded = decode_floorplan(tokenized.tokens, taxonomy)
        errors = validate_vector_floorplan(
            plan,
            taxonomy,
            tolerances=tolerances,
        )
    except ValueError as error:
        return {
            "case_name": batch["case_name"][0],
            "contract_valid": False,
            "error": str(error),
        }
    diagnostics_plan = SimpleNamespace(
        rooms=[
            {"polygon": room.polygon, "holes": room.holes}
            for room in plan.rooms
        ],
        walls=plan.walls,
    )
    watertight = watertight_diagnostics(
        diagnostics_plan,
        taxonomy=taxonomy,
        tolerances=tolerances,
    )
    rasters = render_floorplan_rasters(plan, taxonomy, strict=False)
    return {
        "case_name": batch["case_name"][0],
        "contract_valid": True,
        "room_count": len(plan.rooms),
        "wall_count": len(plan.walls),
        "operation_count": len(plan.operations),
        "token_count": tokenized.token_count,
        "token_bytes": tokenized.byte_count,
        "validation_errors": errors,
        "functional_watertight": float(bool(watertight["watertight"])),
        "hole_count": int(watertight["hole_count"]),
        "dangling_endpoint_rate": float(watertight["dangling_endpoint_rate"]),
        "boundary_collision_pixels": int(
            rasters["room_boundary_collision_pixels"]
        ),
        "outside_footprint_pixels": int(rasters["outside_footprint_pixels"]),
        "plan_token_canonical_exact": (
            canonical_fingerprint(plan) == canonical_fingerprint(tokenized.plan)
        ),
        "token_roundtrip_exact": (
            canonical_fingerprint(tokenized.plan)
            == canonical_fingerprint(decoded)
        ),
    }


__all__ = ["contract_from_output", "contract_sample"]
