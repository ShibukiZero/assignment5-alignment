#!/usr/bin/env python3
"""Compare HF behavior with and without a preinitialized sleeping vLLM backend."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams

from cs336_alignment.backend_lifecycle import BackendLifecycleManager, init_policy
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/sft.jsonl"
DEFAULT_VAL_PATH = (
    "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl"
)
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = ".agents/logs/diagnose_preinitialized_vllm_training_interference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether preinitializing a sleeping vLLM backend perturbs training."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--normalize-constant", type=float, default=1.0)
    parser.add_argument("--num-eval-examples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-level", type=int, default=2)
    parser.add_argument("--sample-dump-count", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_prompt_template(path: str) -> str:
    template = Path(path).read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {path}")
    return template


def get_question(example: dict[str, Any]) -> str:
    question = example.get("question") or example.get("problem")
    if question is None:
        raise ValueError("Example must contain `question` or `problem`.")
    return question


def get_ground_truth(example: dict[str, Any]) -> str:
    ground_truth = example.get("ground_truth", example.get("answer"))
    if ground_truth is None:
        raise ValueError("Example must contain `ground_truth` or `answer`.")
    return str(ground_truth)


def select_examples(
    examples: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [examples[index] for index in indices[:count]]


def iter_minibatches(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start : start + batch_size]]


def trim_after_stop(text: str, stop: str) -> str:
    if stop not in text:
        return text
    prefix, _ = text.split(stop, 1)
    return prefix + stop


def train_one_optimizer_step(
    *,
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Any,
    train_examples: list[dict[str, Any]],
    train_batch_size: int,
    gradient_accumulation_steps: int,
    normalize_constant: float,
    device: str,
    seed: int,
) -> dict[str, float]:
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")

    microbatch_size = train_batch_size // gradient_accumulation_steps
    rng = random.Random(seed)
    train_iter = iter_minibatches(train_examples, microbatch_size, rng)

    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    micro_scaled_losses: list[float] = []

    for _ in range(gradient_accumulation_steps):
        microbatch = next(train_iter)
        prompts = [example["prompt"] for example in microbatch]
        responses = [example["response"] for example in microbatch]
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        scored = get_response_log_probs(
            model=policy,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )
        loss, metadata = sft_microbatch_train_step(
            policy_log_probs=scored["log_probs"],
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=normalize_constant,
        )
        micro_scaled_losses.append(float(loss.detach().cpu().item()))
        micro_losses.append(float(metadata["loss"].detach().cpu().item()))

        del scored, loss, metadata
        del input_ids, labels, response_mask, tokenized

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": mean(micro_losses),
        "scaled_loss": mean(micro_scaled_losses),
        "grad_norm": float(grad_norm.detach().cpu().item()),
    }


def generate_with_hf(
    *,
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[list[str], float]:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0.0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    policy.eval()
    start = time.time()
    with torch.no_grad():
        output_ids = policy.generate(**generate_kwargs)
    elapsed = time.time() - start
    policy.train()

    # With left padding, generated tokens start after the full padded input width,
    # not after the per-example non-pad token count.
    prompt_width = input_ids.shape[1]
    responses: list[str] = []
    for row in output_ids:
        generated_ids = row[prompt_width:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(trim_after_stop(text, "</answer>"))

    tokenizer.padding_side = original_padding_side
    return responses, elapsed


def score_responses(
    *,
    prompts: list[str],
    responses: list[str],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for prompt, response, example in zip(prompts, responses, examples):
        scores = r1_zero_reward_fn(response, get_ground_truth(example))
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": get_ground_truth(example),
                **scores,
            }
        )
    return {
        "records": records,
        "summary": {
            "num_examples": len(records),
            "reward": mean(record["reward"] for record in records),
            "format_accuracy": mean(record["format_reward"] for record in records),
            "answer_accuracy": mean(record["answer_reward"] for record in records),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_training_only_path(
    args: argparse.Namespace,
    *,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    prompt_template: str,
) -> dict[str, Any]:
    policy, tokenizer = init_policy(
        model_id=args.model,
        device=args.policy_device,
    )
    optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
    training_step = train_one_optimizer_step(
        policy=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_examples=train_examples,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        normalize_constant=args.normalize_constant,
        device=args.policy_device,
        seed=args.seed,
    )
    prompts = [prompt_template.format(question=get_question(example)) for example in eval_examples]
    responses, eval_seconds = generate_with_hf(
        policy=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.policy_device,
    )
    scored = score_responses(prompts=prompts, responses=responses, examples=eval_examples)
    scored["summary"]["eval_seconds"] = eval_seconds
    return {
        "training_step": training_step,
        "summary": scored["summary"],
        "records": scored["records"],
    }


def run_preinitialized_sleeping_vllm_path(
    args: argparse.Namespace,
    *,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    prompt_template: str,
) -> dict[str, Any]:
    manager = BackendLifecycleManager.from_defaults(
        model_id=args.model,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        seed=args.seed,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_sleep_mode=True,
        sleep_level=args.sleep_level,
        keep_policy_resident_on_device=True,
        keep_optimizer_state_resident_on_device=False,
    )
    policy, tokenizer, optimizer = manager.initialize_rl_runtime(
        optimizer_factory=lambda parameters: AdamW(parameters, lr=args.learning_rate)
    )
    policy, tokenizer = manager.enter_training_phase()
    training_step = train_one_optimizer_step(
        policy=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_examples=train_examples,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        normalize_constant=args.normalize_constant,
        device=args.policy_device,
        seed=args.seed,
    )
    prompts = [prompt_template.format(question=get_question(example)) for example in eval_examples]
    responses, eval_seconds = generate_with_hf(
        policy=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.policy_device,
    )
    scored = score_responses(prompts=prompts, responses=responses, examples=eval_examples)
    scored["summary"]["eval_seconds"] = eval_seconds
    return {
        "training_step": training_step,
        "summary": scored["summary"],
        "records": scored["records"],
    }


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

    train_examples = read_jsonl(args.train_path)
    val_examples = read_jsonl(args.val_path)
    prompt_template = load_prompt_template(args.prompt_template)
    eval_examples = select_examples(val_examples, count=args.num_eval_examples, seed=args.seed)

    training_only = run_training_only_path(
        args,
        train_examples=train_examples,
        eval_examples=eval_examples,
        prompt_template=prompt_template,
    )
    preinitialized_sleeping_vllm = run_preinitialized_sleeping_vllm_path(
        args,
        train_examples=train_examples,
        eval_examples=eval_examples,
        prompt_template=prompt_template,
    )

    write_json(log_dir / "training_only_training_step_summary.json", training_only["training_step"])
    write_json(log_dir / "training_only_summary.json", training_only["summary"])
    write_jsonl(log_dir / "training_only_records.jsonl", training_only["records"])
    write_jsonl(
        log_dir / "training_only_sample_records.jsonl",
        training_only["records"][: args.sample_dump_count],
    )

    write_json(
        log_dir / "preinitialized_sleeping_vllm_training_step_summary.json",
        preinitialized_sleeping_vllm["training_step"],
    )
    write_json(log_dir / "preinitialized_sleeping_vllm_summary.json", preinitialized_sleeping_vllm["summary"])
    write_jsonl(
        log_dir / "preinitialized_sleeping_vllm_records.jsonl",
        preinitialized_sleeping_vllm["records"],
    )
    write_jsonl(
        log_dir / "preinitialized_sleeping_vllm_sample_records.jsonl",
        preinitialized_sleeping_vllm["records"][: args.sample_dump_count],
    )

    write_json(
        log_dir / "comparison_summary.json",
        {
            "training_only": training_only["summary"],
            "preinitialized_sleeping_vllm": preinitialized_sleeping_vllm["summary"],
            "training_step": {
                "training_only": training_only["training_step"],
                "preinitialized_sleeping_vllm": preinitialized_sleeping_vllm["training_step"],
            },
            "delta": {
                key: preinitialized_sleeping_vllm["summary"][key] - training_only["summary"][key]
                for key in ("reward", "format_accuracy", "answer_accuracy", "eval_seconds")
            },
        },
    )


if __name__ == "__main__":
    main()
