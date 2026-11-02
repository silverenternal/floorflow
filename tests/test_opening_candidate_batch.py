from collections import Counter

import numpy as np
import pytest

from floorflow.data.taxonomy import build_taxonomy
from floorflow.geometry.wall_graph_readout import RasterWallPlan, ReadoutWall
from floorflow.inference.opening_candidate_batch import (
    rank_and_attach_center_width_openings,
)
from floorflow.inference.opening_candidate_ranker import OpeningCandidateRanker


class _ConstantRanker(OpeningCandidateRanker):
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("wall identity must fail before ranking")


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


def _plan(start: float) -> RasterWallPlan:
    return RasterWallPlan(
        frame=64.0,
        walls=[
            ReadoutWall(
                start=(start, 20.0),
                end=(start + 40.0, 20.0),
                thickness=3.0,
            )
        ],
        rooms=[],
        openings=[],
    )


def test_batched_opening_attachment_rejects_wall_identity_mismatch() -> None:
    ranker = OpeningCandidateRanker(
        presence_model=None,
        family_model=None,
        presence_threshold=0.5,
    )
    maps = [np.zeros((64, 64), dtype=np.float32)]
    family = [np.zeros((2, 64, 64), dtype=np.float32)]
    with pytest.raises(ValueError, match="matching host wall identity"):
        rank_and_attach_center_width_openings(
            [_plan(2.0)],
            [_plan(8.0)],
            maps,
            maps,
            maps,
            family,
            _taxonomy(),
            ranker,
            threshold=0.3,
            min_area=3,
            min_peak_distance=4.0,
            frame=64.0,
        )
