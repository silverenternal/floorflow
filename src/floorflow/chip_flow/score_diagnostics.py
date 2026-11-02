"""Per-case components of the official FloorSet score."""

from __future__ import annotations

import numpy as np


def _hpwl_parts(
    boxes: np.ndarray,
    b2b: np.ndarray,
    p2b: np.ndarray,
    pins: np.ndarray,
) -> tuple[float, float]:
    centers = boxes[:, :2] + boxes[:, 2:] / 2.0
    b2b_result = 0.0
    for source, target, weight in np.asarray(b2b).reshape(-1, 3):
        source, target = int(source), int(target)
        if source < 0 or target < 0 or source >= len(boxes) or target >= len(boxes):
            continue
        b2b_result += float(weight) * (
            abs(centers[source, 0] - centers[target, 0])
            + abs(centers[source, 1] - centers[target, 1])
        )
    p2b_result = 0.0
    for pin, block, weight in np.asarray(p2b).reshape(-1, 3):
        pin, block = int(pin), int(block)
        if pin < 0 or block < 0 or pin >= len(pins) or block >= len(boxes):
            continue
        p2b_result += float(weight) * (
            abs(float(pins[pin, 0]) - centers[block, 0])
            + abs(float(pins[pin, 1]) - centers[block, 1])
        )
    return b2b_result, p2b_result


def score_diagnostics(
    prefix: str,
    score,
    boxes: np.ndarray,
    gt_boxes: np.ndarray,
    b2b: np.ndarray,
    p2b: np.ndarray,
    pins: np.ndarray,
) -> dict[str, float | int]:
    """Return components not present in the compact official case rows."""

    predicted_b2b, predicted_p2b = _hpwl_parts(boxes, b2b, p2b, pins)
    target_b2b, target_p2b = _hpwl_parts(gt_boxes, b2b, p2b, pins)
    predicted_rho = np.log(
        np.maximum(boxes[:, 2], 1.0e-6) / np.maximum(boxes[:, 3], 1.0e-6)
    )
    target_rho = np.log(
        np.maximum(gt_boxes[:, 2], 1.0e-6)
        / np.maximum(gt_boxes[:, 3], 1.0e-6)
    )
    soft = score.soft
    feasibility = score.feasibility
    return {
        f"{prefix}_hard_cost": float(score.cost),
        f"{prefix}_area_gap": float(score.area_gap),
        f"{prefix}_hpwl_gap": float(score.hpwl_gap),
        f"{prefix}_b2b_hpwl_gap": float(
            (predicted_b2b - target_b2b) / max(target_b2b, 1.0e-6)
        ),
        f"{prefix}_p2b_hpwl_gap": float(
            (predicted_p2b - target_p2b) / max(target_p2b, 1.0e-6)
        ),
        f"{prefix}_rho_mae": float(np.mean(np.abs(predicted_rho - target_rho))),
        f"{prefix}_v_rel": float(score.v_rel),
        f"{prefix}_grouping": int(soft.grouping) if soft is not None else 0,
        f"{prefix}_mib": int(soft.mib) if soft is not None else 0,
        f"{prefix}_boundary": int(soft.boundary) if soft is not None else 0,
        f"{prefix}_soft_slots": int(soft.n_soft) if soft is not None else 0,
        f"{prefix}_overlap_pairs": int(feasibility.overlap_count),
        f"{prefix}_area_violations": len(feasibility.area_violations),
        f"{prefix}_fixed_violations": len(feasibility.fixed_violations),
        f"{prefix}_preplaced_violations": len(
            feasibility.preplaced_violations
        ),
        f"{prefix}_aspect_violations": len(feasibility.aspect_violations),
    }


__all__ = ["score_diagnostics"]
