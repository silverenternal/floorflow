"""Legalized evaluation for distilled chip candidate consensus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.candidate_consensus_training import (
    CandidateConsensusBatch,
    load_candidate_consensus_batch,
)
from floorflow.chip_flow.candidate_manifest import SOURCE_GEOMETRY_KEYS
from floorflow.chip_flow.hard_constraint_projector import project_hard_constraints
from floorflow.chip_flow.model import decode_boxes_batch


@dataclass(frozen=True)
class CandidateConsensusCase:
    index: int
    batch: CandidateConsensusBatch
    geometry: dict[str, np.ndarray]


def load_candidate_consensus_case(path: Path) -> CandidateConsensusCase:
    """Load a shard and fail closed unless legalized evaluation is possible."""

    payload = np.load(path)
    required = {"case_index", *(f"source_{key}" for key in SOURCE_GEOMETRY_KEYS)}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"candidate consensus case is missing arrays: {sorted(missing)}")
    batch = load_candidate_consensus_batch(path)
    count = int(batch.candidate_states.shape[2])
    geometry = {
        key: np.asarray(payload[f"source_{key}"])
        for key in SOURCE_GEOMETRY_KEYS
    }
    expected = {
        "gt_boxes": (count, 4),
        "areas": (count,),
        "constraints": (count, 5),
        "b2b": (None, 3),
        "p2b": (None, 3),
        "pins": (None, 2),
    }
    for key, shape in expected.items():
        if geometry[key].ndim != len(shape):
            raise ValueError(f"source_{key} has the wrong rank")
        if shape[0] is not None and geometry[key].shape[0] != shape[0]:
            raise ValueError(f"source_{key} has the wrong case length")
        if geometry[key].shape[-1] != shape[-1]:
            raise ValueError(f"source_{key} has the wrong feature width")
    return CandidateConsensusCase(
        index=int(payload["case_index"]),
        batch=batch,
        geometry=geometry,
    )


def _payload(
    case: CandidateConsensusCase,
    boxes: np.ndarray,
    *,
    legalizer_options: dict | None,
    legalizer_time_limit_ms: int,
) -> dict:
    return {
        "index": case.index,
        "blocks": boxes.shape[0],
        "boxes": np.asarray(boxes, dtype=np.float32),
        "gt_boxes": case.geometry["gt_boxes"],
        "areas": case.geometry["areas"],
        "constraints": case.geometry["constraints"],
        "b2b": case.geometry["b2b"],
        "p2b": case.geometry["p2b"],
        "pins": case.geometry["pins"],
        "latent_mse": 0.0,
        "legalize": True,
        "legalizer_time_limit_ms": int(legalizer_time_limit_ms),
        "legalizer_options": dict(legalizer_options or {}),
        "return_raw_boxes": False,
        "return_legalized_boxes": False,
    }


def _project_boxes(
    case: CandidateConsensusCase,
    latent: torch.Tensor,
    *,
    iterations: int,
) -> np.ndarray:
    geometry = case.geometry
    projected = project_hard_constraints(
        latent[None].float(),
        case.batch.target_state.float(),
        torch.as_tensor(geometry["constraints"]).float()[None],
        torch.as_tensor(geometry["areas"]).float()[None],
        iterations=iterations,
    )
    return decode_boxes_batch(
        projected.latent,
        torch.as_tensor(geometry["areas"]).float()[None],
    )[0].detach().cpu().numpy()


def _relative(current: float, reference: float) -> float | None:
    if abs(reference) <= 1.0e-12:
        return None
    return current / reference - 1.0


def evaluate_candidate_consensus_case(
    model: CandidateConsensusFuser,
    case: CandidateConsensusCase,
    scorer: Callable[[dict], dict],
    *,
    projector_iterations: int = 2,
    legalizer_time_limit_ms: int = 2000,
    legalizer_options: dict | None = None,
) -> dict:
    """Decode, project and legalize reference, best-single and fused layouts."""

    model.eval()
    with torch.no_grad():
        fused, weights = model(
            case.batch.candidate_states,
            case.batch.candidate_features,
            case.batch.candidate_mask,
            case.batch.context,
        )
    layouts = {
        f"candidate_{index}": _project_boxes(
            case,
            state,
            iterations=projector_iterations,
        )
        for index, state in enumerate(case.batch.candidate_states[0])
        if bool(case.batch.candidate_mask[0, index])
    }
    layouts["consensus"] = _project_boxes(
        case,
        fused[0],
        iterations=projector_iterations,
    )
    rows = {
        name: scorer(
            _payload(
                case,
                boxes,
                legalizer_options=legalizer_options,
                legalizer_time_limit_ms=legalizer_time_limit_ms,
            )
        )
        for name, boxes in layouts.items()
    }
    valid_names = [name for name in rows if name != "consensus"]
    reference_name = valid_names[0]
    best_name = min(
        valid_names,
        key=lambda name: float(rows[name]["legalized_hard_cost"]),
    )
    metrics = (
        "legalized_hard_cost",
        "legalized_v_rel",
        "legalized_area_gap",
        "legalized_b2b_hpwl_gap",
        "legalized_p2b_hpwl_gap",
        "legalized_grouping",
        "legalized_boundary",
        "legalized_rho_mae",
    )
    comparisons = {}
    for baseline_name in (reference_name, best_name):
        comparisons[baseline_name] = {
            metric: _relative(
                float(rows["consensus"][metric]),
                float(rows[baseline_name][metric]),
            )
            for metric in metrics
        }
    return {
        "index": case.index,
        "case_index": case.index,
        "candidate_count": len(valid_names),
        "weights": weights[0].detach().cpu().tolist(),
        "reference": reference_name,
        "best_single": best_name,
        "rows": rows,
        "consensus_vs_reference": comparisons[reference_name],
        "consensus_vs_best_single": comparisons[best_name],
    }


def evaluate_legalized_candidates(
    case: CandidateConsensusCase,
    scorer: Callable[[dict], dict],
    *,
    projector_iterations: int = 2,
    legalizer_time_limit_ms: int = 2000,
    legalizer_options: dict | None = None,
) -> dict:
    """Score every real candidate without adding a synthetic consensus arm."""

    rows = {}
    for index, state in enumerate(case.batch.candidate_states[0]):
        if not bool(case.batch.candidate_mask[0, index]):
            continue
        boxes = _project_boxes(
            case,
            state,
            iterations=projector_iterations,
        )
        rows[f"candidate_{index}"] = scorer(
            _payload(
                case,
                boxes,
                legalizer_options=legalizer_options,
                legalizer_time_limit_ms=legalizer_time_limit_ms,
            )
        )
    return {
        "index": case.index,
        "case_index": case.index,
        "candidate_count": len(rows),
        "rows": rows,
    }


__all__ = [
    "CandidateConsensusCase",
    "evaluate_candidate_consensus_case",
    "evaluate_legalized_candidates",
    "load_candidate_consensus_case",
]
