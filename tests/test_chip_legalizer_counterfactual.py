from __future__ import annotations

from floorflow.chip_flow.legalizer_counterfactual import (
    compare_legalizer_policies,
    policies_from_config,
)
from floorflow.chip_flow.legalizer_policy import LegalizerPolicy


def test_legalizer_counterfactual_reuses_identical_case_payload() -> None:
    seen = []

    def score(payload: dict) -> dict:
        seen.append(payload)
        options = payload["legalizer_options"]
        scale = 1.0 if options["reshape"] else 1.1
        return {
            "legalized_hard_cost": scale,
            "legalized_v_rel": scale,
            "legalized_area_gap": scale,
            "legalized_b2b_hpwl_gap": scale,
            "legalized_p2b_hpwl_gap": scale,
        }

    row = compare_legalizer_policies(
        {"index": 7, "boxes": [[0.0, 0.0, 1.0, 1.0]]},
        {
            "default": LegalizerPolicy(),
            "shape_preserving": LegalizerPolicy(reshape=False),
        },
        score,
    )
    assert len(seen) == 2
    assert seen[0]["boxes"] == seen[1]["boxes"]
    assert row["variants"]["default"]["relative_to_reference"][
        "legalized_hard_cost"
    ] == 0.0
    assert row["variants"]["shape_preserving"]["relative_to_reference"][
        "legalized_hard_cost"
    ] > 0.0


def test_legalizer_counterfactual_parses_config_variants() -> None:
    policies = policies_from_config(
        {
            "variants": [
                {
                    "id": "default",
                    "relation": "restored",
                    "reshape": True,
                    "trust": 1.3,
                },
                {
                    "id": "low_trust",
                    "relation": "expanded",
                    "reshape": False,
                    "trust": 1.1,
                },
            ]
        }
    )
    assert set(policies) == {"default", "low_trust"}
    assert policies["low_trust"].relation == "expanded"
