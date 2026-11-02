from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.opening_readout import OpeningDetection
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutWall
from floorflow.inference.opening_candidate_ranker import (
    FEATURE_NAMES,
    FEATURE_NAMES_V3,
    HOST_FEATURE_NAMES,
    OpeningCandidateRanker,
    load_opening_candidate_ranker,
    opening_candidate_features,
    opening_candidate_features_v3,
    save_opening_candidate_ranker,
)


class _FixedModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        size = len(features)
        return np.tile(np.asarray([[0.2, 0.8]]), (size, 1))


class _FamilyModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        size = len(features)
        return np.tile(np.asarray([[0.25, 0.75]]), (size, 1))


def _taxonomy():
    return build_taxonomy(
        {
            "room": Counter({"Bedroom": 1}),
            "furniture": Counter(),
            "door": Counter({"Swing": 1}),
            "window": Counter({"Regular": 1}),
        },
        room_min_count=1,
        furniture_min_count=1,
    )


def _fixture():
    taxonomy = _taxonomy()
    plan = RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(
                start=(4.0, 20.0),
                end=(60.0, 20.0),
                thickness=3.0,
                wall_type="exterior",
            )
        ],
        rooms=[],
        openings=[],
    )
    detections = [
        OpeningDetection(
            center=(20.0, 20.0),
            width=8.0,
            family="door",
            class_id=taxonomy.opening_ids[0],
            score=0.8,
            wall_index=0,
            wall_distance=0.0,
        ),
        OpeningDetection(
            center=(40.0, 20.0),
            width=12.0,
            family="window",
            class_id=taxonomy.opening_ids[1],
            score=0.6,
            wall_index=0,
            wall_distance=0.0,
        ),
    ]
    foreground = np.zeros((64, 64), dtype=np.float32)
    family_logits = np.zeros((2, 64, 64), dtype=np.float32)
    family_logits[0, 18:23, 16:25] = 2.0
    family_logits[1, 18:23, 34:47] = 2.0
    return taxonomy, plan, detections, foreground, family_logits


def test_opening_candidate_features_use_wall_interval_context() -> None:
    taxonomy, plan, detections, foreground, family_logits = _fixture()
    features = opening_candidate_features(
        detections[0],
        foreground,
        family_logits,
        plan,
        detections,
        frame=64.0,
    )
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert features[FEATURE_NAMES.index("wall_length_fraction")] > 0.8
    assert features[FEATURE_NAMES.index("exterior_wall")] == 1.0
    assert features[FEATURE_NAMES.index("same_wall_candidate_count")] == 2.0
    assert taxonomy.opening_ids


def test_duplicate_equal_candidates_keep_distinct_identity_features() -> None:
    _taxonomy_value, plan, detections, foreground, family_logits = _fixture()
    duplicate = replace(detections[0])
    candidates = [detections[0], duplicate]
    features = opening_candidate_features(
        duplicate,
        foreground,
        family_logits,
        plan,
        candidates,
        frame=64.0,
    )
    assert features[FEATURE_NAMES.index("same_wall_score_rank")] == 0.5
    assert features[FEATURE_NAMES.index("previous_center_gap_fraction")] == 0.0
    assert features[FEATURE_NAMES.index("next_center_gap_fraction")] > 0.0


def test_opening_candidate_ranker_filters_and_retypes() -> None:
    taxonomy, plan, detections, foreground, family_logits = _fixture()
    ranker = OpeningCandidateRanker(
        presence_model=_FixedModel(),
        family_model=_FamilyModel(),
        presence_threshold=0.75,
    )
    ranked = ranker.rank(
        detections,
        foreground,
        family_logits,
        plan,
        taxonomy,
        frame=64.0,
    )
    assert len(ranked) == 2
    assert {item.family for item in ranked} == {"window"}


def test_host_feature_contract_adds_visibility_and_fragmentation() -> None:
    taxonomy, plan, detections, foreground, family_logits = _fixture()
    features = opening_candidate_features_v3(
        detections[0],
        foreground,
        family_logits,
        plan,
        detections,
        frame=64.0,
    )
    assert features.shape == (len(FEATURE_NAMES_V3),)
    assert features.shape[0] == len(FEATURE_NAMES) + len(HOST_FEATURE_NAMES)
    assert np.isfinite(features).all()
    assert (
        features[FEATURE_NAMES_V3.index("host_wall_continuity")] == 1.0
    )
    assert taxonomy.opening_ids


def test_v3_ranker_contract_round_trips(tmp_path) -> None:
    ranker = OpeningCandidateRanker(
        presence_model=_FixedModel(),
        family_model=_FamilyModel(),
        presence_threshold=0.75,
        feature_names=FEATURE_NAMES_V3,
    )
    path = tmp_path / "ranker.joblib"
    save_opening_candidate_ranker(path, ranker)
    loaded = load_opening_candidate_ranker(path)
    assert loaded.feature_names == FEATURE_NAMES_V3

