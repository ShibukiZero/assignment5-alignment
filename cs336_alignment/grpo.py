from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Score rollouts and compute group-relative advantages."""
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length.")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("The number of rollout responses must be divisible by group_size.")
    if normalize_by_std and group_size == 1:
        raise ValueError("normalize_by_std=True requires group_size > 1.")

    raw_reward_values = [
        float(reward_fn(response, ground_truth)["reward"])
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw_reward_values, dtype=torch.float32)

    # Rollouts are ordered as contiguous groups: one question, then its G responses.
    grouped_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=-1, keepdim=True)
    advantages = grouped_rewards - group_means
    if normalize_by_std:
        # Use torch.std's default sample standard deviation; this matches the tests.
        group_stds = grouped_rewards.std(dim=-1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)

    flat_advantages = advantages.reshape(-1)
    reward_std = raw_rewards.std().item() if raw_rewards.numel() > 1 else 0.0
    advantage_std = flat_advantages.std().item() if flat_advantages.numel() > 1 else 0.0
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": reward_std,
        "min_reward": raw_rewards.min().item(),
        "max_reward": raw_rewards.max().item(),
        "mean_advantage": flat_advantages.mean().item(),
        "std_advantage": advantage_std,
        "num_groups": float(grouped_rewards.shape[0]),
        "group_size": float(group_size),
    }
    return flat_advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """Compute the per-token REINFORCE-style policy-gradient loss."""
    if raw_rewards_or_advantages.ndim != 2 or raw_rewards_or_advantages.shape[-1] != 1:
        raise ValueError("raw_rewards_or_advantages must have shape (batch_size, 1).")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")
    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("raw_rewards_or_advantages and policy_log_probs batch sizes must match.")

    return -raw_rewards_or_advantages * policy_log_probs
