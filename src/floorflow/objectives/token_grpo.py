"""Group-relative policy optimization for generated floorplan tokens."""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.shape[0] % int(group_size):
        raise ValueError("rewards must form complete groups")
    grouped = rewards.reshape(-1, int(group_size))
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / std.clamp_min(1.0e-6)).reshape(-1)


def token_grpo_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    valid: torch.Tensor,
    rewards: torch.Tensor,
    *,
    group_size: int,
    reference_logits: torch.Tensor | None = None,
    kl_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits and target ids must align")
    if valid.shape != target_ids.shape or rewards.shape[0] != logits.shape[0]:
        raise ValueError("masks and rewards must align with logits")
    advantages = group_relative_advantages(rewards.detach(), group_size)
    log_probability = logits.log_softmax(dim=-1).gather(
        -1,
        target_ids.clamp(0, logits.shape[-1] - 1)[..., None],
    ).squeeze(-1)
    sequence_log_probability = (
        log_probability * valid
    ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    policy_loss = -(
        advantages.to(logits.dtype) * sequence_log_probability
    ).mean()
    kl = logits.new_zeros(())
    if reference_logits is not None and kl_weight > 0.0:
        policy_log_probability = logits.log_softmax(dim=-1)
        reference_probability = reference_logits.softmax(dim=-1)
        token_kl = (
            reference_probability
            * (reference_probability.log() - policy_log_probability)
        ).sum(dim=-1)
        kl = (token_kl * valid).sum() / valid.sum().clamp_min(1)
    total = policy_loss + float(kl_weight) * kl
    return total, {
        "token_grpo_loss": policy_loss.detach(),
        "token_grpo_kl": kl.detach(),
        "token_grpo_reward_mean": rewards.mean().detach(),
        "token_grpo_advantage_abs_mean": advantages.abs().mean().detach(),
    }


__all__ = ["group_relative_advantages", "token_grpo_loss"]
