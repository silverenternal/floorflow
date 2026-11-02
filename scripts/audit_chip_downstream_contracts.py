"""CPU-only audit of projector, refiner and legalizer-response contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRINITY_SOURCE = ROOT / "third_party/Trinity/src"
if str(TRINITY_SOURCE) not in sys.path:
    sys.path.insert(0, str(TRINITY_SOURCE))

from floorflow.chip_flow.cache import PackedCaseStore
from floorflow.chip_flow.endpoint_compatibility import (
    endpoint_compatibility_terms,
)
from floorflow.chip_flow.hard_constraint_projector import (
    project_hard_constraints,
)
from floorflow.chip_flow.legalizer_response import (
    LegalizerResponseHead,
    legalizer_response_from_scale_pack,
)
from floorflow.chip_flow.model import decode_boxes_batch
from floorflow.chip_flow.projector_geometry import candidate_overlap_pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/cache/chip_floorset_50k_train_12k_dev_v2/dev"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/chip_downstream_contract_audit_v1.json"),
    )
    parser.add_argument("--cases", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    store = PackedCaseStore(args.cache)
    block_count = next(
        int(count)
        for count, indices in sorted(store.by_block_count.items())
        if len(indices) >= args.cases
    )
    indices = store.indices(block_count)[: args.cases]
    batch = store.batch(indices, device="cpu")
    source = batch["source_latent"].clone().requires_grad_(True)
    boxes = decode_boxes_batch(source.detach(), batch["areas"])
    dense_pairs = block_count * (block_count - 1) // 2
    candidate_pairs = candidate_overlap_pairs(boxes)[0].numel()
    sparse_output = project_hard_constraints(
        source,
        batch["target_latent"],
        batch["constraints"],
        batch["areas"],
        iterations=2,
        sparse_pairs=True,
        projection_strategy="legacy",
    )
    conflict_output = project_hard_constraints(
        source,
        batch["target_latent"],
        batch["constraints"],
        batch["areas"],
        iterations=2,
        sparse_pairs=True,
        projection_strategy="conflict_aware",
    )
    sparse_output.latent.square().mean().backward()
    projector_grad_finite = bool(torch.isfinite(source.grad).all())

    trajectory_source = batch["source_latent"].clone().requires_grad_(True)
    compatibility = endpoint_compatibility_terms(
        trajectory_source,
        batch,
        iterations=1,
        refiner_steps=2,
    )
    compatibility.total.backward()
    refiner_grad_finite = bool(torch.isfinite(trajectory_source.grad).all())

    target_boxes = batch["boxes"]
    legalized_boxes = target_boxes.clone()
    legalized_boxes[..., :2] += 0.01
    legalizer_targets = legalizer_response_from_scale_pack(
        target_boxes,
        legalized_boxes,
        batch["areas"],
        {"rung": 2},
    )
    response_head = LegalizerResponseHead(32, bottleneck=8)
    report = {
        "version": "chip-downstream-contract-audit-v1",
        "status": "pass"
        if projector_grad_finite and refiner_grad_finite
        else "fail",
        "device": "cpu",
        "cache": str(args.cache),
        "block_count": block_count,
        "cases": args.cases,
        "overlap_candidates": {
            "dense_upper_triangle_pairs": dense_pairs,
            "grid_filtered_pairs": int(candidate_pairs),
            "reduction": 1.0 - float(candidate_pairs) / max(1, dense_pairs),
        },
        "projector": {
            "legacy_overlap_pairs": sparse_output.remaining_overlap_pairs,
            "conflict_aware_overlap_pairs": (
                conflict_output.remaining_overlap_pairs
            ),
            "legacy_boundary_violations": sparse_output.boundary_violations,
            "conflict_aware_boundary_violations": (
                conflict_output.boundary_violations
            ),
            "legacy_joint_energy": sparse_output.joint_conflict_energy,
            "conflict_aware_joint_energy": (
                conflict_output.joint_conflict_energy
            ),
            "autograd_finite": projector_grad_finite,
            "out_of_place": True,
        },
        "refiner_trajectory": {
            "steps": 2,
            "autograd_finite": refiner_grad_finite,
            "energy": float(compatibility.refiner_trajectory_energy.detach()),
            "displacement": float(
                compatibility.refiner_trajectory_displacement.detach()
            ),
        },
        "legalizer_response": {
            "head_parameters": sum(
                parameter.numel() for parameter in response_head.parameters()
            ),
            "displacement_shape": list(
                legalizer_targets["legalizer_displacement"].shape
            ),
            "rung_shape": list(legalizer_targets["legalizer_rung"].shape),
            "target_source": "scale_pack xywh plus last_info rung",
        },
        "boundary": "CPU contract audit only; no quality, GPU or ORFS claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
