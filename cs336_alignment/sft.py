from __future__ import annotations

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
