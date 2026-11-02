from __future__ import annotations

from floorflow.chip_flow.candidate_subset_audit import (
    budget_matched_subsets,
    validation_rank,
)


def test_budget_matched_subsets_keep_reference_and_fixed_budget() -> None:
    subsets = budget_matched_subsets(5, budget=4, required=(0,))
    assert subsets == [
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 3, 4),
        (0, 2, 3, 4),
    ]


def test_validation_rank_prioritizes_gate_coverage_then_wins() -> None:
    def result(safe: float, strict: float, joint: float) -> dict:
        return {
            "validation": {
                "versus_anchor": {
                    "wins": {
                        "constraint_safe_rate": safe,
                        "strict_structural_rate": strict,
                        "joint_rate": joint,
                    },
                    "relative": {
                        "legalized_hard_cost": 0.0,
                        "legalized_v_rel": 0.0,
                        "legalized_grouping": 0.0,
                        "legalized_rho_mae": 0.0,
                    },
                }
            }
        }

    passing = result(0.36, 0.40, 0.70)
    mean_only = result(0.30, 0.50, 0.80)
    assert validation_rank(passing) > validation_rank(mean_only)
