"""Matched counterfactual runner over frozen refined chip layouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from floorflow.chip_flow.legalizer_policy import LegalizerPolicy


def compare_legalizer_policies(
    payload: dict,
    variants: Mapping[str, LegalizerPolicy],
    score_case: Callable[[dict], dict],
) -> dict:
    """Score every policy on the exact same refined layout."""

    if not variants:
        raise ValueError("legalizer comparison requires at least one policy")
    rows = {}
    for name, policy in variants.items():
        policy.validate()
        candidate = dict(payload)
        candidate["legalize"] = True
        candidate["legalizer_options"] = policy.kwargs()
        candidate["legalizer_time_limit_ms"] = policy.time_limit_ms
        rows[name] = score_case(candidate)
    reference_name = "default" if "default" in rows else next(iter(rows))
    reference = rows[reference_name]
    for name, row in rows.items():
        row["relative_to_reference"] = {
            key: (
                float(row[key]) - float(reference[key])
            ) / max(abs(float(reference[key])), 1.0e-9)
            for key in (
                "legalized_hard_cost",
                "legalized_v_rel",
                "legalized_area_gap",
                "legalized_b2b_hpwl_gap",
                "legalized_p2b_hpwl_gap",
            )
            if key in row and key in reference
        }
        row["policy"] = name
    return {
        "index": int(payload["index"]),
        "reference": reference_name,
        "variants": rows,
    }


def policies_from_config(config: Mapping) -> dict[str, LegalizerPolicy]:
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("legalizer config requires variants")
    policies = {}
    for variant in variants:
        name = str(variant["id"])
        if name in policies:
            raise ValueError(f"duplicate legalizer variant: {name}")
        policies[name] = LegalizerPolicy(
            relation=str(variant.get("relation", "restored")),
            reshape=bool(variant.get("reshape", True)),
            trust=float(variant.get("trust", 1.3)),
            outline=bool(variant.get("outline", True)),
            time_limit_ms=int(variant.get("time_limit_ms", 2000)),
        )
    return policies


__all__ = ["compare_legalizer_policies", "policies_from_config"]
