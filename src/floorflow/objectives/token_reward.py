"""Deterministic reward for SFT-to-GRPO token post-training."""

from __future__ import annotations


def token_candidate_reward(
    contract: dict[str, object],
    syntax: dict[str, float],
    *,
    expected_room_count: int,
) -> tuple[float, dict[str, float]]:
    valid = float(bool(contract.get("contract_valid", False)))
    watertight = float(contract.get("functional_watertight", 0.0))
    holes = float(contract.get("hole_count", 0.0))
    room_error = abs(
        int(contract.get("room_count", expected_room_count)) - expected_room_count
    )
    public_advantage = float(syntax.get("public_integration_advantage", 0.0))
    reward = (
        2.0 * valid
        + 3.0 * watertight
        - 0.2 * holes
        - 0.2 * room_error
        + 0.5 * public_advantage
    )
    return reward, {
        "reward_validity": valid,
        "reward_watertight": watertight,
        "reward_holes": holes,
        "reward_room_error": float(room_error),
        "reward_public_advantage": public_advantage,
    }


__all__ = ["token_candidate_reward"]
