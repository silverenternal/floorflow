"""Audit cacheability, family separation, and bridge behavior of CFM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.distribution.cfm_audit import (
    condition_cache_reuse_ratio,
    family_stream_distinctness,
)
from floorflow.distribution.checkpoint import restore_distribution_model
from floorflow.model.cfm_structure_adapter import enable_cfm_structure

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/train/distribution_joint_full_v8_budgetfix/best.pt"
        ),
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--solvers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/distribution_cfm_representation_audit_v1.json",
    )
    args = parser.parse_args()
    device = torch.device("cpu")
    model, taxonomy, _payload = restore_distribution_model(
        args.checkpoint,
        device,
        bridge_mode="residual",
    )
    enable_cfm_structure(model.base, taxonomy)
    model.eval()
    footprint = torch.zeros(1, 1, 64, 64)
    footprint[:, :, 8:56, 8:56] = 1.0
    condition = model.encode_condition(footprint)
    state = model.contour_initial_state(
        footprint,
        condition_features=condition,
    )
    cfm_state = state["cfm"]["state"]
    streams = model.base.family_stream_evidence(
        cfm_state,
        condition[-1],
    )
    nfe = max(1, int(args.steps) * int(args.solvers))
    report = {
        "schema_version": "distribution-cfm-representation-audit-v1",
        "checkpoint": str(args.checkpoint),
        "bridge_mode": model.bridge_mode,
        "static_condition_encoder": model.base.static_condition_encoder,
        "typed_family_streams": model.base.typed_family_streams,
        "multiscale_structure_injection": (
            model.base.multiscale_structure_injection
        ),
        "condition_cache_reuse_ratio": condition_cache_reuse_ratio(nfe),
        "family_stream_distinctness": family_stream_distinctness(streams),
        "state_outside_norm": float(
            cfm_state[:, :, ~footprint[0, 0].bool()]
            .detach()
            .abs()
            .mean()
        ),
        "claim_boundary": (
            "Architecture audit only; no generation-quality claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
