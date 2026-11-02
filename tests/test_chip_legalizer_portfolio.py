from __future__ import annotations

from floorflow.chip_flow.legalizer_portfolio import (
    LegalizerCandidate,
    select_legalizer_candidate,
)


def _candidate(
    name: str,
    v_rel: float,
    *,
    rung: int = 2,
    solves: int = 4,
    b2b_hpwl: float | None = None,
    p2b_hpwl: float | None = None,
    outline_compactness: float | None = None,
    outline_aspect: float | None = None,
) -> LegalizerCandidate:
    return LegalizerCandidate(
        name=name,
        v_rel=v_rel,
        displacement=1.0,
        boundary=1.0,
        grouping=1.0,
        rung=rung,
        solves=solves,
        b2b_hpwl=b2b_hpwl,
        p2b_hpwl=p2b_hpwl,
        outline_compactness=outline_compactness,
        outline_aspect=outline_aspect,
    )


def test_legalizer_portfolio_chooses_lower_v_rel_at_same_solver_work() -> None:
    baseline = _candidate("baseline", 0.2)
    candidate = _candidate("candidate", 0.1)
    assert select_legalizer_candidate(
        baseline,
        (candidate,),
    ).name == "candidate"


def test_legalizer_portfolio_rejects_solver_escalation() -> None:
    baseline = _candidate("baseline", 0.2)
    candidate = _candidate("candidate", 0.1, rung=3)
    assert select_legalizer_candidate(
        baseline,
        (candidate,),
    ).name == "baseline"


def test_geometry_consensus_rejects_a_catastrophic_low_vrel_outlier() -> None:
    baseline = _candidate(
        "baseline",
        0.2,
        b2b_hpwl=2.0,
        p2b_hpwl=2.0,
        outline_compactness=0.5,
        outline_aspect=-1.0,
    )
    low_vrel_outlier = _candidate(
        "low_vrel_outlier",
        0.1,
        b2b_hpwl=4.0,
        p2b_hpwl=4.0,
        outline_compactness=0.2,
        outline_aspect=-1.2,
    )
    balanced = _candidate(
        "balanced",
        0.11,
        b2b_hpwl=1.0,
        p2b_hpwl=1.0,
        outline_compactness=0.9,
        outline_aspect=-0.1,
    )
    assert select_legalizer_candidate(
        baseline,
        (low_vrel_outlier, balanced),
        policy="geometry_consensus",
    ).name == "balanced"


def test_geometry_guard_preserves_legality_first_until_geometry_is_risky() -> None:
    baseline = _candidate(
        "baseline",
        0.2,
        b2b_hpwl=2.0,
        p2b_hpwl=2.0,
        outline_compactness=0.9,
        outline_aspect=-0.2,
    )
    risky = _candidate(
        "risky",
        0.1,
        b2b_hpwl=4.0,
        p2b_hpwl=4.0,
        outline_compactness=0.3,
        outline_aspect=-1.2,
    )
    safe = _candidate(
        "safe",
        0.11,
        b2b_hpwl=1.0,
        p2b_hpwl=1.0,
        outline_compactness=0.95,
        outline_aspect=-0.1,
    )
    assert select_legalizer_candidate(
        baseline,
        (risky, safe),
        policy="legality_guarded_consensus",
    ).name == "safe"
