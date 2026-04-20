from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from cs336_alignment.sft import masked_normalize


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


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO clipped policy-gradient loss.

    This is the PPO-style clipped surrogate used by GRPO. The ratio compares
    the current policy with the rollout policy that produced the sampled tokens.
    """
    if advantages.ndim != 2 or advantages.shape[-1] != 1:
        raise ValueError("advantages must have shape (batch_size, 1).")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs must have the same shape as policy_log_probs.")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages and policy_log_probs batch sizes must match.")
    if cliprange < 0:
        raise ValueError("cliprange must be non-negative.")

    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, min=1.0 - cliprange, max=1.0 + cliprange)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    per_token_loss = -torch.minimum(unclipped_objective, clipped_objective)

    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "is_clipped": (ratio != clipped_ratio),
    }
    return per_token_loss, metadata


def compute_grpo_no_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the unclipped GRPO surrogate used for the clip ablation."""
    if advantages.ndim != 2 or advantages.shape[-1] != 1:
        raise ValueError("advantages must have shape (batch_size, 1).")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("old_log_probs must have the same shape as policy_log_probs.")
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages and policy_log_probs batch sizes must match.")

    ratio = torch.exp(policy_log_probs - old_log_probs)
    per_token_loss = -(ratio * advantages)
    metadata = {
        "ratio": ratio,
    }
    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatch to the requested per-token policy-gradient loss."""
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        return loss, {}

    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    if loss_type == "grpo_no_clip":
        return compute_grpo_no_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
        )

    raise ValueError(f"Unknown policy-gradient loss_type: {loss_type}")


def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    """Average tensor values over positions where mask is true."""
    if tensor.shape != mask.shape:
        raise ValueError("tensor and mask must have the same shape.")

    float_mask = mask.to(dtype=tensor.dtype)
    masked_sum = (tensor * float_mask).sum(dim=dim)
    mask_count = float_mask.sum(dim=dim)
    return masked_sum / mask_count


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    loss_normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate one GRPO microbatch loss."""
    if response_mask.shape != policy_log_probs.shape:
        raise ValueError("response_mask must have the same shape as policy_log_probs.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if loss_normalization == "masked_normalize" and loss_normalize_constant <= 0:
        raise ValueError("loss_normalize_constant must be positive.")

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'.")
        loss_raw_rewards = raw_rewards
        loss_advantages = raw_rewards
        loss_old_log_probs = policy_log_probs
        loss_cliprange = 0.0
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'.")
        loss_raw_rewards = advantages
        loss_advantages = advantages
        loss_old_log_probs = policy_log_probs
        loss_cliprange = 0.0
    elif loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='grpo_clip'.")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type='grpo_clip'.")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type='grpo_clip'.")
        loss_raw_rewards = raw_rewards if raw_rewards is not None else advantages
        loss_advantages = advantages
        loss_old_log_probs = old_log_probs
        loss_cliprange = cliprange
    elif loss_type == "grpo_no_clip":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='grpo_no_clip'.")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for loss_type='grpo_no_clip'.")
        loss_raw_rewards = raw_rewards if raw_rewards is not None else advantages
        loss_advantages = advantages
        loss_old_log_probs = old_log_probs
        loss_cliprange = 0.0
    else:
        raise ValueError(f"Unknown policy-gradient loss_type: {loss_type}")

    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=loss_raw_rewards,
        advantages=loss_advantages,
        old_log_probs=loss_old_log_probs,
        cliprange=loss_cliprange,
    )
    if loss_normalization == "masked_mean":
        per_example_loss = masked_mean(tensor=per_token_loss, mask=response_mask, dim=-1)
    elif loss_normalization == "masked_normalize":
        per_example_loss = masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            dim=-1,
            normalize_constant=loss_normalize_constant,
        )
    else:
        raise ValueError(f"Unknown loss_normalization: {loss_normalization}")

    loss = per_example_loss.mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = dict(metadata)
    metadata["loss"] = loss.detach()
    return scaled_loss, metadata
