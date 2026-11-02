"""Audit directional behavior of training-only differentiable FEVP proxies."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from floorflow.objectives.differentiable_fevp import (
    differentiable_fevp_terms,
    soft_opening_attachment,
    soft_outside_preservation,
    soft_room_overlap,
    soft_wall_continuity,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/residential_differentiable_fevp_audit_v1.json"


def main() -> None:
    footprint = torch.ones(1, 32, 32)
    room_low = torch.zeros(1, 2, 32, 32, requires_grad=True)
    with torch.no_grad():
        room_low[:, 0] = 0.7
        room_low[:, 1] = 0.1
    room_high = room_low.detach().clone().requires_grad_(True)
    with torch.no_grad():
        room_high[:, 1] = 0.7
    overlap_low = soft_room_overlap(room_low, footprint)
    overlap_high = soft_room_overlap(room_high, footprint)

    wall_continuous = torch.zeros(1, 32, 32)
    wall_continuous[:, :, 15:17] = 1.0
    wall_gap = wall_continuous.clone()
    wall_gap[:, 12:20, 15:17] = 0.0
    seam_rooms = torch.zeros(1, 2, 32, 32)
    seam_rooms[:, 0, :, :16] = 1.0
    seam_rooms[:, 1, :, 16:] = 1.0
    wall_clean = soft_wall_continuity(seam_rooms, wall_continuous, footprint)
    wall_broken = soft_wall_continuity(seam_rooms, wall_gap, footprint)

    opening = torch.zeros(1, 32, 32)
    opening[:, 2:4, 2:4] = 1.0
    opening_attached = soft_opening_attachment(
        torch.zeros_like(opening),
        torch.ones_like(opening),
        footprint,
    )
    opening_detached = soft_opening_attachment(
        opening,
        torch.zeros_like(opening),
        footprint,
    )

    before = torch.zeros(1, 2, 32, 32)
    before[:, 0] = 1.0
    after = before.clone()
    after[:, 0, :, 2] = 0.0
    edit = torch.zeros(1, 32, 32)
    edit[:, 8:, :] = 1.0
    preservation = soft_outside_preservation(
        after,
        before,
        footprint,
        edit,
    )

    after_total, terms = differentiable_fevp_terms(
        room_high,
        room_low.detach(),
        wall_gap,
        opening,
        footprint,
        edit,
    )
    after_total.backward()
    checks = {
        "overlap_direction": bool(overlap_high > overlap_low),
        "wall_continuity_direction": bool(wall_broken > wall_clean),
        "opening_attachment_direction": bool(
            opening_detached > opening_attached
        ),
        "outside_preservation_direction": bool(preservation > 0),
        "combined_loss_finite": bool(torch.isfinite(after_total)),
        "all_terms_finite": all(
            bool(torch.isfinite(value)) for value in terms.values()
        ),
        "gradient_finite": (
            room_high.grad is not None
            and bool(torch.isfinite(room_high.grad).all())
        ),
        "soft_names_only": all(name.startswith("soft_") for name in terms),
    }
    report = {
        "schema_version": "residential-differentiable-fevp-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "training_proxy_only",
        "not_fevp_metrics": True,
        "checks": checks,
        "direction_values": {
            "overlap_low": float(overlap_low.detach()),
            "overlap_high": float(overlap_high.detach()),
            "wall_clean": float(wall_clean.detach()),
            "wall_broken": float(wall_broken.detach()),
            "opening_attached": float(opening_attached.detach()),
            "opening_detached": float(opening_detached.detach()),
            "outside_preservation": float(preservation.detach()),
        },
        "terms": {key: float(value) for key, value in terms.items()},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "checks": checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
