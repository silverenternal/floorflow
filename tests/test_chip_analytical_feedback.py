import json
from pathlib import Path

import pytest

from floorflow.chip_flow.analytical_feedback import (
    seed_portfolio_diagnostic,
    summarize_analytical_response,
    summarize_matched_flow_search,
)

ROOT = Path(__file__).resolve().parents[1]


def test_analytical_response_uses_globalplace_wirelength() -> None:
    result = summarize_analytical_response(
        {
            "globalplace__route__wirelength__estimated": 12.5,
            "globalplace__gpl__routability__congestion": 0.4,
            "globalplace__timing__setup__ws": -0.1,
            "globalplace__timing__setup__tns": -2.0,
            "globalplace__design__instance__utilization__stdcell": 0.6,
        }
    )

    assert result["score"] == pytest.approx(12.5)
    assert result["score_name"] == "globalplace__route__wirelength__estimated"
    assert result["missing"] == []
    assert "detailed-route wirelength" in result["forbidden_inputs"]


def test_analytical_response_reports_missing_score() -> None:
    result = summarize_analytical_response({})

    assert result["score"] is None
    assert "globalplace__route__wirelength__estimated" in result["missing"]


def test_analytical_response_rejects_negative_score() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_analytical_response(
            {"globalplace__route__wirelength__estimated": -1.0}
        )


def test_seed_portfolio_selects_lowest_globalplace_score(tmp_path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "inventory": [
                    {
                        "design_key": "p/design",
                        "platform": "p",
                        "design_nickname": "design",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    comparisons = []
    for seed, wirelength in ((19, 100.0), (20, 90.0)):
        comparisons.append(
            {
                "case": "p/design",
                "seed": seed,
                "status": "completed",
                "detailedroute__route__wirelength": {
                    "candidate": wirelength,
                    "control": 100.0,
                },
            }
        )
        stage = (
            tmp_path
            / "orfs"
            / "logs"
            / "p"
            / "design"
            / f"ppa_holdout_v44_candidate_s{seed}"
            / "3_3_place_gp.json"
        )
        stage.parent.mkdir(parents=True)
        stage.write_text(
            json.dumps(
                {
                    "globalplace__route__wirelength__estimated": (
                        200.0 if seed == 19 else 150.0
                    )
                }
            ),
            encoding="utf-8",
        )
    ppa = tmp_path / "ppa.json"
    ppa.write_text(json.dumps({"comparisons": comparisons}), encoding="utf-8")

    result = seed_portfolio_diagnostic(ppa, inventory, tmp_path / "orfs")

    assert result["designs"][0]["selected_seed"] == 20
    assert result["means"]["selected_relative_to_same_seed_control"] == pytest.approx(
        -0.1
    )


def test_analytical_search_selects_lowest_declared_score() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "evaluate_chip_analytical_search",
        ROOT / "scripts/evaluate_chip_analytical_search.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = [
        {
            "proposal": 0,
            "seed": 19,
            "p2b_hpwl_um": 2.0,
            "globalplace_wirelength": 8.0,
            "placement_tcl": "p0.tcl",
            "placement_report": "p0.json",
            "analytical_report": "a0.json",
        },
        {
            "proposal": 1,
            "seed": 1019,
            "p2b_hpwl_um": 3.0,
            "globalplace_wirelength": 7.0,
            "placement_tcl": "p1.tcl",
            "placement_report": "p1.json",
            "analytical_report": "a1.json",
        },
    ]

    assert module.select(rows, "globalplace_wirelength")["proposal"] == 1
    assert module.select(rows, "p2b_hpwl_um")["proposal"] == 0


def _search_report(
    *,
    arm: str,
    scores: list[float | None],
    status: str = "generated",
) -> dict:
    return {
        "status": status,
        "arm": arm,
        "proposal_count": len(scores),
        "results": [
            {
                "case": "p/design",
                "proposals": [
                    {
                        "proposal": index,
                        "seed": 19 + 1000 * index,
                        "nfe": 4 // len(scores),
                        "refiner_steps": 1600 // len(scores),
                        "p2b_hpwl_um": 10.0,
                        "globalplace_wirelength": score,
                        "placement_report": f"{arm}_p{index}.json",
                        "placement_tcl": f"{arm}_p{index}.tcl",
                        "analytical_report": f"{arm}_a{index}.json",
                    }
                    for index, score in enumerate(scores)
                ],
            }
        ],
    }


def test_matched_flow_summary_resolves_p2b_tie() -> None:
    reports = {
        "candidate_n1": _search_report(arm="candidate", scores=[100.0]),
        "control_n1": _search_report(arm="control", scores=[110.0]),
        "candidate_n2": _search_report(arm="candidate", scores=[103.0, 99.0]),
        "control_n2": _search_report(arm="control", scores=[108.0, 109.0]),
    }
    result = summarize_matched_flow_search(reports)

    assert result["status"] == "complete"
    candidate_n2 = result["arms"]["p/design|candidate|n2"]
    assert candidate_n2["p2b"]["is_tie"]
    assert candidate_n2["analytical"]["proposal"] == 1
    assert candidate_n2["selection_agreement"] == "analytical_resolves_p2b_tie"
    comparison = result["candidate_control_comparisons"][1]
    assert comparison["analytical_selected_relative_to_control"] == pytest.approx(
        99.0 / 108.0 - 1.0
    )


def test_matched_flow_summary_keeps_missing_response_pending() -> None:
    reports = {
        "candidate_n1": _search_report(arm="candidate", scores=[100.0]),
        "control_n1": _search_report(arm="control", scores=[110.0]),
        "candidate_n2": _search_report(arm="candidate", scores=[None, 99.0]),
        "control_n2": _search_report(arm="control", scores=[108.0, None]),
    }
    result = summarize_matched_flow_search(reports)

    assert result["status"] == "pending_analytical_response"
    assert result["arms"]["p/design|candidate|n2"]["selection_agreement"] == "pending"


def test_matched_flow_summary_accepts_completed_response_reports() -> None:
    reports = {
        "candidate_n1": _search_report(
            arm="candidate",
            scores=[100.0],
            status="complete",
        ),
        "control_n1": _search_report(
            arm="control",
            scores=[110.0],
            status="complete",
        ),
        "candidate_n2": _search_report(
            arm="candidate",
            scores=[103.0, 99.0],
            status="complete",
        ),
        "control_n2": _search_report(
            arm="control",
            scores=[108.0, 109.0],
            status="complete",
        ),
    }
    result = summarize_matched_flow_search(reports)
    assert result["status"] == "complete"
