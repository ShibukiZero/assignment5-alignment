from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and mask response tokens in shifted labels."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    tokenized_examples: list[tuple[list[int], list[int], list[int]]] = []
    max_prompt_and_output_length = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        prompt_and_output_ids = prompt_ids + output_ids
        if len(prompt_and_output_ids) < 2:
            raise ValueError("Each prompt/output pair must tokenize to at least two tokens.")

        tokenized_examples.append((prompt_ids, output_ids, prompt_and_output_ids))
        max_prompt_and_output_length = max(max_prompt_and_output_length, len(prompt_and_output_ids))

    input_ids_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []
    response_mask_batch: list[list[int]] = []
    for prompt_ids, output_ids, prompt_and_output_ids in tokenized_examples:
        padding_length = max_prompt_and_output_length - len(prompt_and_output_ids)
        padded_prompt_and_output_ids = prompt_and_output_ids + [pad_token_id] * padding_length
        input_ids = padded_prompt_and_output_ids[:-1]
        labels = padded_prompt_and_output_ids[1:]

        response_start_in_labels = max(len(prompt_ids) - 1, 0)
        response_mask = [0] * len(labels)
        response_end = min(response_start_in_labels + len(output_ids), len(labels))
        response_mask[response_start_in_labels:response_end] = [1] * (
            response_end - response_start_in_labels
        )

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        response_mask_batch.append(response_mask)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_batch, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute next-token entropy over the vocabulary dimension."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """Get per-token log-probabilities for the provided next-token labels."""
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    label_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    output = {"log_probs": label_log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    """Sum masked tensor values and divide by a fixed normalization constant."""
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate the masked SFT negative log-likelihood for one microbatch."""
    if normalize_constant is None:
        normalize_constant = response_mask.sum().item()

    per_example_loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    )
    loss = per_example_loss.mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, {"loss": loss.detach()}


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_strs: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    generation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate responses and return per-example logs plus aggregate metrics."""
    if len(prompt_strs) != len(ground_truths):
        raise ValueError("prompt_strs and ground_truths must have the same length.")

    generation_kwargs = dict(generation_kwargs or {})
    if "max_new_tokens" not in generation_kwargs:
        generation_kwargs["max_new_tokens"] = 1024
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        generation_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

    device = next(model.parameters()).device
    encoded_prompts = tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**encoded_prompts, **generation_kwargs)

    prompt_width = encoded_prompts["input_ids"].shape[-1]
    output_strs: list[str] = []
    response_lengths: list[int] = []
    for generated in generated_ids:
        response_ids = generated[prompt_width:]
        if tokenizer.eos_token_id is not None:
            non_eos_response_ids = response_ids[response_ids != tokenizer.eos_token_id]
        else:
            non_eos_response_ids = response_ids
        if tokenizer.pad_token_id is not None:
            non_eos_response_ids = non_eos_response_ids[
                non_eos_response_ids != tokenizer.pad_token_id
            ]
        response_lengths.append(int(non_eos_response_ids.numel()))
        output_strs.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    with torch.no_grad():
        scored = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=True,
        )

    if was_training:
        model.train()

    token_entropy = scored["token_entropy"]
    per_example_entropy = masked_normalize(
        tensor=token_entropy,
        mask=response_mask,
        dim=-1,
        normalize_constant=1.0,
    ) / response_mask.sum(dim=-1).clamp(min=1)

    examples: list[dict[str, Any]] = []
    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    entropy_values: list[float] = []
    correct_lengths: list[float] = []
    incorrect_lengths: list[float] = []

    for i, (prompt, response, ground_truth) in enumerate(
        zip(prompt_strs, output_strs, ground_truths)
    ):
        reward_info = reward_fn(response, ground_truth)
        reward = float(reward_info.get("reward", 0.0))
        format_reward = float(reward_info.get("format_reward", 0.0))
        answer_reward = float(reward_info.get("answer_reward", 0.0))
        avg_entropy = float(per_example_entropy[i].detach().cpu().item())
        response_length = float(response_lengths[i])

        rewards.append(reward)
        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
        entropy_values.append(avg_entropy)
        if answer_reward == 1.0:
            correct_lengths.append(response_length)
        else:
            incorrect_lengths.append(response_length)

        examples.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": reward,
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "avg_token_entropy": avg_entropy,
                "response_length": response_lengths[i],
            }
        )

    return {
        "examples": examples,
        "summary": {
            "num_examples": len(examples),
            "avg_reward": _mean_or_none(rewards),
            "format_accuracy": _mean_or_none(format_rewards),
            "answer_accuracy": _mean_or_none(answer_rewards),
            "avg_token_entropy": _mean_or_none(entropy_values),
            "avg_response_length": _mean_or_none([float(x) for x in response_lengths]),
            "avg_correct_response_length": _mean_or_none(correct_lengths),
            "avg_incorrect_response_length": _mean_or_none(incorrect_lengths),
        },
    }
