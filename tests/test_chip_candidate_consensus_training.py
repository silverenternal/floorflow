from __future__ import annotations

import numpy as np
import pytest
import torch

from floorflow.chip_flow.candidate_consensus import CandidateConsensusFuser
from floorflow.chip_flow.candidate_consensus_evaluation import (
    evaluate_candidate_consensus_case,
    evaluate_legalized_candidates,
    load_candidate_consensus_case,
)
from floorflow.chip_flow.candidate_consensus_training import (
    CandidateConsensusBatch,
    candidate_consensus_objective,
    load_candidate_consensus_batch,
    load_candidate_consensus_checkpoint,
    write_candidate_consensus_cohort,
)


def _batch() -> CandidateConsensusBatch:
    states = torch.randn(2, 3, 4, 3)
    states[1, 2] = float("nan")
    features = torch.randn(2, 3, 5)
    features[1, 2] = float("nan")
    mask = torch.tensor([[True, True, True], [True, True, False]])
    return CandidateConsensusBatch(
        candidate_states=states,
        candidate_features=features,
        quality=torch.tensor([[1.2, 1.1, 1.0], [1.5, 1.2, 9.0]]),
        target_state=torch.randn(2, 4, 3),
        candidate_mask=mask,
        context=torch.randn(2, 2),
    )


def test_candidate_consensus_objective_is_trainable_and_masks_invalid_candidates() -> None:
    torch.manual_seed(51)
    model = CandidateConsensusFuser(
        hidden=16,
        candidate_feature_dim=5,
        context_dim=2,
    )
    model.gate.data.fill_(1.0)
    loss, weights, terms = candidate_consensus_objective(model, _batch())
    assert torch.isfinite(loss)
    assert torch.allclose(weights[1, 2], torch.tensor(0.0))
    loss.backward()
    assert model.scorer[-1].weight.grad is not None
    assert terms.endpoint >= 0
    assert terms.teacher >= 0


def test_candidate_consensus_loader_validates_required_arrays(tmp_path) -> None:
    path = tmp_path / "cohort.npz"
    np.savez(
        path,
        candidate_states=np.random.randn(1, 2, 3, 3).astype("float32"),
        candidate_features=np.random.randn(1, 2, 4).astype("float32"),
        quality=np.asarray([[1.0, 0.9]], dtype="float32"),
        target_state=np.random.randn(1, 3, 3).astype("float32"),
        candidate_mask=np.asarray([[True, True]]),
    )
    batch = load_candidate_consensus_batch(path)
    assert batch.context is None
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, candidate_states=np.zeros((1, 2, 3, 3)))
    with pytest.raises(ValueError, match="missing arrays"):
        load_candidate_consensus_batch(bad_path)


def test_candidate_consensus_rejects_nonfinite_valid_teacher_quality() -> None:
    batch = _batch()
    batch.quality[0, 0] = float("nan")
    with pytest.raises(ValueError, match="quality must be finite"):
        batch.validate()


def test_candidate_consensus_checkpoint_loader_rebuilds_dimensions(tmp_path) -> None:
    model = CandidateConsensusFuser(
        hidden=12,
        candidate_feature_dim=5,
        context_dim=2,
    )
    path = tmp_path / "consensus.pt"
    torch.save(
        {
            "config": {
                "hidden": 12,
                "feature_dim": 5,
                "context_dim": 2,
            },
            "model": model.state_dict(),
        },
        path,
    )
    restored = load_candidate_consensus_checkpoint(path)
    assert restored.hidden == 12
    assert restored.candidate_feature_dim == 5
    assert restored.context_dim == 2


def test_sparse_selection_is_bounded_fallback_safe_and_trainable() -> None:
    torch.manual_seed(53)
    model = CandidateConsensusFuser(
        hidden=16,
        candidate_feature_dim=5,
        selection_mode="topk_residual",
        top_k=2,
        residual_limit=0.02,
    )
    states = torch.randn(2, 3, 4, 3)
    features = torch.randn(2, 3, 5)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    with torch.no_grad():
        model.gate.fill_(1.0)
    fused, weights = model(states, features, mask)
    assert torch.allclose(weights[1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.all((weights >= 0.0) & (weights <= 1.0))
    selected = torch.einsum("bk,bknd->bnd", weights.detach(), states)
    assert torch.all((fused - selected).abs() <= 0.020001)
    fused.square().mean().backward()
    assert model.ranker is not None
    assert any(parameter.grad is not None for parameter in model.ranker.parameters())


def test_sparse_selection_checkpoint_restores_mode_and_limit(tmp_path) -> None:
    model = CandidateConsensusFuser(
        hidden=12,
        candidate_feature_dim=5,
        selection_mode="topk_residual",
        top_k=1,
        residual_limit=0.03,
    )
    path = tmp_path / "selection.pt"
    torch.save(
        {
            "config": {
                "hidden": 12,
                "feature_dim": 5,
                "selection_mode": "topk_residual",
                "top_k": 1,
                "residual_limit": 0.03,
            },
            "model": model.state_dict(),
        },
        path,
    )
    restored = load_candidate_consensus_checkpoint(path)
    assert restored.selection_mode == "topk_residual"
    assert restored.top_k == 1
    assert restored.residual_limit == 0.03


def test_candidate_consensus_cohort_writer_round_trips_case_shards(tmp_path) -> None:
    cases = [
        {
            "index": 17,
            "target_state": np.zeros((3, 3)).tolist(),
            "source_geometry": {
                "areas": [1.0, 1.0, 1.0],
                "constraints": np.zeros((3, 5)).tolist(),
            },
            "context": [0.1, 0.2],
            "candidates": [
                {
                    "state": np.ones((3, 3)).tolist(),
                    "features": [0.1, 0.2],
                    "quality": 1.2,
                },
                {
                    "state": np.full((3, 3), 0.5).tolist(),
                    "features": [0.3, 0.4],
                    "quality": 1.0,
                },
            ],
        }
    ]
    paths = write_candidate_consensus_cohort(tmp_path, cases)
    assert len(paths) == 1
    saved = np.load(paths[0])
    assert int(saved["case_index"]) == 17
    assert saved["source_areas"].shape == (3,)
    batch = load_candidate_consensus_batch(paths[0])
    assert batch.candidate_states.shape == (1, 2, 3, 3)
    assert batch.context is not None
    assert batch.context.shape == (1, 2)
    with pytest.raises(ValueError, match="indices must be unique"):
        write_candidate_consensus_cohort(tmp_path, cases + cases)


def test_candidate_consensus_case_reaches_existing_legalized_scorer(tmp_path) -> None:
    cases = [
        {
            "index": 17,
            "target_state": np.zeros((2, 3)).tolist(),
            "source_geometry": {
                "gt_boxes": [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
                "areas": [1.0, 1.0],
                "constraints": np.zeros((2, 5)).tolist(),
                "b2b": [[0.0, 1.0, 1.0]],
                "p2b": [[0.0, 0.0, 1.0]],
                "pins": [[0.0, 0.0]],
            },
            "candidates": [
                {
                    "state": np.zeros((2, 3)).tolist(),
                    "features": [0.0, 0.0],
                    "quality": 1.1,
                },
                {
                    "state": np.zeros((2, 3)).tolist(),
                    "features": [1.0, 1.0],
                    "quality": 1.0,
                },
            ],
        }
    ]
    path = write_candidate_consensus_cohort(tmp_path, cases)[0]
    case = load_candidate_consensus_case(path)
    model = CandidateConsensusFuser(
        hidden=8,
        candidate_feature_dim=2,
    )

    def scorer(payload: dict) -> dict:
        score = 1.0 + float(np.asarray(payload["boxes"]).mean())
        return {
            "legalized_hard_cost": score,
            "legalized_v_rel": score / 10.0,
            "legalized_area_gap": score / 20.0,
            "legalized_b2b_hpwl_gap": score / 30.0,
            "legalized_p2b_hpwl_gap": score / 40.0,
            "legalized_grouping": score / 50.0,
            "legalized_boundary": score / 60.0,
            "legalized_rho_mae": score / 70.0,
        }

    result = evaluate_candidate_consensus_case(model, case, scorer)
    assert result["case_index"] == 17
    assert result["candidate_count"] == 2
    assert result["consensus_vs_reference"]["legalized_hard_cost"] == 0.0


def test_legalized_candidate_labels_exclude_consensus(tmp_path) -> None:
    cases = [
        {
            "index": 23,
            "target_state": np.zeros((2, 3)).tolist(),
            "source_geometry": {
                "gt_boxes": [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
                "areas": [1.0, 1.0],
                "constraints": np.zeros((2, 5)).tolist(),
                "b2b": [[0.0, 1.0, 1.0]],
                "p2b": [[0.0, 0.0, 1.0]],
                "pins": [[0.0, 0.0]],
            },
            "candidates": [
                {
                    "state": np.zeros((2, 3)).tolist(),
                    "features": [0.0, 0.0],
                    "quality": 1.1,
                },
                {
                    "state": np.ones((2, 3)).tolist(),
                    "features": [1.0, 1.0],
                    "quality": 1.0,
                },
            ],
        }
    ]
    case = load_candidate_consensus_case(
        write_candidate_consensus_cohort(tmp_path, cases)[0]
    )

    def scorer(payload: dict) -> dict:
        value = float(np.asarray(payload["boxes"]).mean())
        return {
            "legalized_hard_cost": value,
            "legalized_grouping": value,
            "legalized_rho_mae": value,
        }

    result = evaluate_legalized_candidates(case, scorer)
    assert result["case_index"] == 23
    assert result["candidate_count"] == 2
    assert set(result["rows"]) == {"candidate_0", "candidate_1"}
