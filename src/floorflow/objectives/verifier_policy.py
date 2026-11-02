"""Policy-gradient feedback from verified state rewards."""

from __future__ import annotations

import torch


def verifier_policy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid: torch.Tensor,
    rewards: torch.Tensor,
    *,
    baseline: float = 0.0,
    reward_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits and target ids must align")
    if rewards.shape[0] != logits.shape[0]:
        raise ValueError("one reward is required per sequence")
    if reward_mask is None:
        reward_mask = torch.ones_like(rewards, dtype=torch.bool)
    if reward_mask.shape != rewards.shape or reward_mask.dtype != torch.bool:
        raise ValueError("reward_mask must be boolean and match rewards")
    log_probability = logits.log_softmax(dim=-1).gather(
        -1,
        target_ids.clamp(0, logits.shape[-1] - 1)[..., None],
    ).squeeze(-1)
    sequence_log_probability = (
        log_probability * valid
    ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    advantages = (rewards.detach() - float(baseline)).to(logits.dtype)
    selected = reward_mask.to(logits.dtype)
    loss = -(
        advantages * sequence_log_probability * selected
    ).sum() / selected.sum().clamp_min(1.0)
    return loss, {
        "verifier_policy_loss": loss.detach(),
        "verifier_reward_mean": (
            (rewards.detach() * selected).sum()
            / selected.sum().clamp_min(1.0)
        ),
        "verifier_advantage_abs_mean": (
            (advantages.abs() * selected).sum()
            / selected.sum().clamp_min(1.0)
        ),
        "verifier_reward_coverage": selected.mean().detach(),
    }


__all__ = ["verifier_policy_loss"]
