from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from cs336_alignment.prompt_templates import format_alpaca_sft_prompt


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _sequence_log_prob(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> Tensor:
    text = format_alpaca_sft_prompt(instruction=prompt, response=response).rstrip()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    if len(token_ids) < 2:
        raise ValueError("Formatted prompt/response must contain at least two tokens.")

    device = _model_device(model)
    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits
    token_log_probs = torch.gather(
        F.log_softmax(logits.float(), dim=-1),
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return token_log_probs.sum()


def preference_log_ratio(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    chosen_log_prob = _sequence_log_prob(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        response=response_chosen,
    )
    rejected_log_prob = _sequence_log_prob(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        response=response_rejected,
    )
    return chosen_log_prob - rejected_log_prob


def per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    policy_log_ratio = preference_log_ratio(
        model=lm,
        tokenizer=tokenizer,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
    )
    with torch.no_grad():
        ref_log_ratio = preference_log_ratio(
            model=lm_ref,
            tokenizer=tokenizer,
            prompt=prompt,
            response_chosen=response_chosen,
            response_rejected=response_rejected,
        ).to(policy_log_ratio.device)

    logits = beta * (policy_log_ratio - ref_log_ratio)
    return -F.logsigmoid(logits)


def per_instance_dpo_loss_with_metrics(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    policy_log_ratio = preference_log_ratio(
        model=lm,
        tokenizer=tokenizer,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
    )
    with torch.no_grad():
        ref_log_ratio = preference_log_ratio(
            model=lm_ref,
            tokenizer=tokenizer,
            prompt=prompt,
            response_chosen=response_chosen,
            response_rejected=response_rejected,
        ).to(policy_log_ratio.device)

    implicit_reward_margin = policy_log_ratio - ref_log_ratio
    logits = beta * implicit_reward_margin
    loss = -F.logsigmoid(logits)
    return loss, {
        "policy_log_ratio": policy_log_ratio.detach(),
        "ref_log_ratio": ref_log_ratio.detach(),
        "dpo_margin": logits.detach(),
        "classification_correct": (implicit_reward_margin.detach() > 0).to(torch.float32),
    }


def dpo_classification_correct(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> bool:
    with torch.no_grad():
        return bool(
            preference_log_ratio(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response_chosen=response_chosen,
                response_rejected=response_rejected,
            )
            > 0
        )
