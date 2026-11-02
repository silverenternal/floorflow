"""Deployable analytical-placement feedback for chip proposal ranking."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ANALYTICAL_RESPONSE_KEYS = (
    "globalplace__route__wirelength__estimated",
    "globalplace__gpl__routability__congestion",
    "globalplace__timing__setup__ws",
    "globalplace__timing__setup__tns",
    "globalplace__design__instance__utilization__stdcell",
)


def summarize_analytical_response(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a deterministic, label-free response from a global-place stage."""

    values = {key: metrics.get(key) for key in ANALYTICAL_RESPONSE_KEYS}
    missing = [
        key for key, value in values.items() if value in (None, "N/A", "ERR")
    ]
    wirelength = values["globalplace__route__wirelength__estimated"]
    score = (
        float(wirelength)
        if wirelength not in (None, "N/A", "ERR")
        else None
    )
    if score is not None and (not math.isfinite(score) or score < 0.0):
        raise ValueError("global-place estimated wirelength must be finite and non-negative")
    return {
        "values": values,
        "missing": missing,
        "score_name": "globalplace__route__wirelength__estimated",
        "score": score,
        "lower_is_better": True,
        "deployment_inputs": "netlist and a legalized macro placement",
        "forbidden_inputs": [
            "detailed-route wirelength",
            "detailed-route DRC",
            "post-route timing",
            "ORFS-oracle case selection",
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "n": len(x),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _selection(
    rows: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any] | None:
    available = [
        row
        for row in rows
        if row.get(score_key) is not None
        and math.isfinite(float(row[score_key]))
    ]
    if not available:
        return None
    selected = min(available, key=lambda row: float(row[score_key]))
    score = float(selected[score_key])
    tolerance = 1.0e-9 * max(1.0, abs(score))
    tied = [
        int(row["proposal"])
        for row in available
        if abs(float(row[score_key]) - score) <= tolerance
    ]
    return {
        "proposal": int(selected["proposal"]),
        "seed": int(selected["seed"]),
        "score": score,
        "available_count": len(available),
        "tied_proposals": tied,
        "is_tie": len(tied) > 1,
        "placement_report": selected.get("placement_report"),
        "placement_tcl": selected.get("placement_tcl"),
        "analytical_report": selected.get("analytical_report"),
    }


def _selection_agreement(
    p2b: dict[str, Any] | None,
    analytical: dict[str, Any] | None,
    expected_count: int,
) -> str:
    if (
        p2b is None
        or analytical is None
        or analytical["available_count"] < expected_count
    ):
        return "pending"
    if p2b["is_tie"]:
        if analytical["is_tie"]:
            return "both_tied"
        if analytical["proposal"] in p2b["tied_proposals"]:
            return "analytical_resolves_p2b_tie"
        return "disagree"
    if analytical["is_tie"]:
        return (
            "analytical_tie_includes_p2b_choice"
            if p2b["proposal"] in analytical["tied_proposals"]
            else "disagree"
        )
    return (
        "agree"
        if p2b["proposal"] == analytical["proposal"]
        else "disagree"
    )


def _relative(candidate: float, control: float) -> float:
    return candidate / control - 1.0


def summarize_matched_flow_search(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    total_nfe: int = 4,
    total_refiner_steps: int = 1600,
) -> dict[str, Any]:
    """Summarize matched candidate/control analytical-search arm reports."""

    by_case: dict[tuple[str, str, int], dict[str, Any]] = {}
    arm_checks: dict[str, dict[str, Any]] = {}
    for arm, report in reports.items():
        arm_name = str(report.get("arm", arm))
        proposal_count = int(report.get("proposal_count", 0))
        expected_nfe = total_nfe // proposal_count if proposal_count else None
        expected_refiner = (
            total_refiner_steps // proposal_count if proposal_count else None
        )
        rows = [
            {**proposal, "case": case["case"]}
            for case in report.get("results", [])
            for proposal in case.get("proposals", [])
        ]
        available = [
            row for row in rows if row.get("globalplace_wirelength") is not None
        ]
        arm_checks[arm] = {
            "report_status": report.get("status"),
            "proposal_count": len(rows),
            "available_analytical_scores": len(available),
            "nfe_contract": all(
                int(row.get("nfe", -1)) == expected_nfe for row in rows
            ),
            "refiner_contract": all(
                int(row.get("refiner_steps", -1)) == expected_refiner
                for row in rows
            ),
        }
        for case in report.get("results", []):
            proposals = case.get("proposals", [])
            p2b = _selection(proposals, "p2b_hpwl_um")
            analytical = _selection(proposals, "globalplace_wirelength")
            by_case[(case["case"], arm_name, proposal_count)] = {
                "p2b": p2b,
                "analytical": analytical,
                "selection_agreement": _selection_agreement(
                    p2b,
                    analytical,
                    len(proposals),
                ),
                "generation_seconds": sum(
                    float(
                        (row.get("generation_run") or {}).get("elapsed_seconds")
                        or 0.0
                    )
                    for row in proposals
                ),
                "response_seconds": sum(
                    float(
                        (row.get("response_run") or {}).get("elapsed_seconds")
                        or 0.0
                    )
                    for row in proposals
                ),
            }
    comparisons = []
    cases = sorted({key[0] for key in by_case})
    for case in cases:
        for proposal_count in sorted({key[2] for key in by_case}):
            candidate = by_case.get((case, "candidate", proposal_count))
            control = by_case.get((case, "control", proposal_count))
            if candidate is None or control is None:
                continue
            analytical = candidate["analytical"]
            control_analytical = control["analytical"]
            comparisons.append(
                {
                    "case": case,
                    "proposal_count": proposal_count,
                    "candidate": candidate,
                    "control": control,
                    "analytical_selected_relative_to_control": (
                        _relative(
                            analytical["score"],
                            control_analytical["score"],
                        )
                        if analytical is not None and control_analytical is not None
                        else None
                    ),
                }
            )
    budget_fixed_search = []
    for arm in sorted({key[1] for key in by_case}):
        for case in cases:
            n1 = by_case.get((case, arm, 1))
            n2 = by_case.get((case, arm, 2))
            if n1 is None or n2 is None:
                continue
            budget_fixed_search.append(
                {
                    "case": case,
                    "arm": arm,
                    "n1_analytical": n1["analytical"],
                    "n2_analytical": n2["analytical"],
                    "n2_relative_to_n1": (
                        _relative(n2["analytical"]["score"], n1["analytical"]["score"])
                        if n1["analytical"] is not None
                        and n2["analytical"] is not None
                        else None
                    ),
                }
            )
    all_rows = [
        row
        for report in reports.values()
        for case in report.get("results", [])
        for row in case.get("proposals", [])
    ]
    all_complete = all(
        row.get("globalplace_wirelength") is not None for row in all_rows
    )
    contracts = all(
        row["report_status"] in {"generated", "complete"}
        and row["nfe_contract"]
        and row["refiner_contract"]
        for row in arm_checks.values()
    )
    return {
        "version": "chip-v46-matched-flow-search-summary-v1",
        "status": (
            "complete"
            if all_complete and contracts
            else "pending_analytical_response"
            if contracts
            else "reject_contract"
        ),
        "budget": {
            "total_nfe": total_nfe,
            "total_refiner_steps": total_refiner_steps,
            "arm_checks": arm_checks,
        },
        "arms": {
            f"{case}|{arm}|n{count}": summary
            for (case, arm, count), summary in sorted(by_case.items())
        },
        "candidate_control_comparisons": comparisons,
        "n1_vs_n2": budget_fixed_search,
        "claim_boundary": (
            "This is a three-case mechanism pilot. Routed PPA, congestion, DRC "
            "and timing of the selected proposals remain independent endpoint gates."
        ),
    }


def seed_portfolio_diagnostic(
    ppa_path: Path,
    inventory_path: Path,
    orfs_work_home: Path,
) -> dict[str, Any]:
    """Measure retrospective seed selection with global-place feedback."""

    ppa = _load(ppa_path)
    inventory = {
        row["design_key"]: row for row in _load(inventory_path)["inventory"]
    }
    comparisons = {
        (row["case"], int(row["seed"])): row
        for row in ppa["comparisons"]
        if row["status"] == "completed"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted({row["case"] for row in ppa["comparisons"]}):
        inventory_row = inventory[case]
        for seed in (19, 20, 21):
            usage = comparisons.get((case, seed))
            stage_path = (
                orfs_work_home
                / "logs"
                / inventory_row["platform"]
                / inventory_row["design_nickname"]
                / f"ppa_holdout_v44_candidate_s{seed}"
                / "3_3_place_gp.json"
            )
            if usage is None or not stage_path.is_file():
                continue
            wirelength = usage["detailedroute__route__wirelength"]
            grouped[case].append(
                {
                    "case": case,
                    "seed": seed,
                    "globalplace_wirelength": float(
                        _load(stage_path)[
                            "globalplace__route__wirelength__estimated"
                        ]
                    ),
                    "candidate_routed_wirelength": float(wirelength["candidate"]),
                    "control_routed_wirelength": float(wirelength["control"]),
                }
            )
    designs = []
    normalized_signal = []
    normalized_target = []
    for case, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        base = next(
            (row for row in rows if row["seed"] == 19),
            min(rows, key=lambda row: row["seed"]),
        )
        selected = min(rows, key=lambda row: row["globalplace_wirelength"])
        oracle = min(rows, key=lambda row: row["candidate_routed_wirelength"])
        for row in rows:
            normalized_signal.append(
                row["globalplace_wirelength"] / base["globalplace_wirelength"] - 1.0
            )
            normalized_target.append(
                row["candidate_routed_wirelength"]
                / base["candidate_routed_wirelength"]
                - 1.0
            )
        designs.append(
            {
                "case": case,
                "available_seeds": [row["seed"] for row in rows],
                "base_seed": base["seed"],
                "selected_seed": selected["seed"],
                "oracle_seed": oracle["seed"],
                "base_relative_to_base_control": (
                    base["candidate_routed_wirelength"]
                    / base["control_routed_wirelength"]
                    - 1.0
                ),
                "selected_relative_to_base_control": (
                    selected["candidate_routed_wirelength"]
                    / base["control_routed_wirelength"]
                    - 1.0
                ),
                "selected_relative_to_same_seed_control": (
                    selected["candidate_routed_wirelength"]
                    / selected["control_routed_wirelength"]
                    - 1.0
                ),
                "oracle_relative_to_base_control": (
                    oracle["candidate_routed_wirelength"]
                    / base["control_routed_wirelength"]
                    - 1.0
                ),
            }
        )
    return {
        "source": str(ppa_path),
        "designs_with_multiple_seeds": len(designs),
        "normalized_candidate_rank": _rank(normalized_signal, normalized_target),
        "designs": designs,
        "means": {
            "base_relative_to_base_control": float(
                np.mean([row["base_relative_to_base_control"] for row in designs])
            ),
            "selected_relative_to_base_control": float(
                np.mean(
                    [row["selected_relative_to_base_control"] for row in designs]
                )
            ),
            "selected_relative_to_same_seed_control": float(
                np.mean(
                    [
                        row["selected_relative_to_same_seed_control"]
                        for row in designs
                    ]
                )
            ),
            "oracle_relative_to_base_control": float(
                np.mean([row["oracle_relative_to_base_control"] for row in designs])
            ),
        },
        "boundary": (
            "Retrospective portfolio diagnostic over already observed seeds. "
            "It is not an equal-budget result because every candidate retains "
            "the full refiner budget, and it cannot confirm a new mechanism."
        ),
    }


__all__ = [
    "ANALYTICAL_RESPONSE_KEYS",
    "seed_portfolio_diagnostic",
    "summarize_analytical_response",
    "summarize_matched_flow_search",
]
