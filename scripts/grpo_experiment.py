#!/usr/bin/env python3
"""Run GRPO on MATH-style reasoning tasks."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output

from sft_experiment import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    append_jsonl,
    cuda_memory_metrics,
    get_ground_truth,
    get_question,
    init_policy,
    init_vllm,
    load_policy_into_vllm_instance,
    load_prompt_template,
    read_jsonl,
    write_json,
)


logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/train.jsonl"
DEFAULT_VAL_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl"
DEFAULT_LOG_DIR = ".agents/logs/ch7/grpo_on_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO on MATH reasoning tasks.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for large outputs such as rollout dumps and optional checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for small logs, metrics, configs, and summaries.",
    )
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
        help="Total number of rollout responses per GRPO step.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of sampled responses per question.",
    )
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--epochs-per-rollout-batch",
        type=int,
        default=1,
        help="Number of training epochs to run over each rollout batch.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Number of rollout responses used for each optimizer update.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument(
        "--loss-type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
        default="reinforce_with_baseline",
    )
    parser.add_argument(
        "--cliprange",
        type=float,
        default=0.2,
        help="PPO/GRPO clipping range used with --loss-type grpo_clip.",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=["masked_mean", "masked_normalize"],
        default="masked_mean",
        help="How to aggregate per-token policy-gradient loss into a per-example loss.",
    )
    parser.add_argument(
        "--loss-normalize-constant",
        type=float,
        default=None,
        help=(
            "Fixed denominator for --loss-normalization masked_normalize. "
            "Defaults to --max-new-tokens."
        ),
    )
    parser.add_argument("--use-std-normalization", dest="use_std_normalization", action="store_true")
    parser.add_argument(
        "--no-use-std-normalization",
        dest="use_std_normalization",
        action="store_false",
    )
    parser.set_defaults(use_std_normalization=True)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run validation every this many GRPO steps.",
    )
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=1024,
        help="Maximum validation examples. Use 0 or a negative value for the full set.",
    )
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--max-train-questions", type=int, default=None)
    parser.add_argument("--sample-rollouts-to-log", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_policy and final_policy under --output-dir.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_question_batch(
    examples: list[dict[str, Any]],
    num_questions: int,
    rng: random.Random,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(examples))
    if num_questions >= len(indexed):
        rng.shuffle(indexed)
        return indexed
    return rng.sample(indexed, num_questions)


def build_rollout_records(
    question_batch: list[tuple[int, dict[str, Any]]],
    prompt_template: str,
    outputs: list[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (question_index, example), request_output in zip(question_batch, outputs):
        question = get_question(example)
        prompt = prompt_template.format(question=question)
        ground_truth = get_ground_truth(example)
        for rollout_index, completion in enumerate(request_output.outputs):
            response = completion.text
            scores = r1_zero_reward_fn(response, ground_truth)
            token_ids = getattr(completion, "token_ids", None)
            records.append(
                {
                    "question_index": question_index,
                    "rollout_index": rollout_index,
                    "question": question,
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": ground_truth,
                    "response_token_length": len(token_ids) if token_ids is not None else None,
                    **scores,
                }
            )
    return records


def generate_rollouts(
    llm: LLM,
    question_batch: list[tuple[int, dict[str, Any]]],
    prompt_template: str,
    group_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    prompts = [
        prompt_template.format(question=get_question(example))
        for _, example in question_batch
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
        n=group_size,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    return build_rollout_records(
        question_batch=question_batch,
        prompt_template=prompt_template,
        outputs=outputs,
    )


def attach_advantages(
    records: list[dict[str, Any]],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    score_cache = {
        (record["response"], record["ground_truth"]): {
            "reward": record["reward"],
            "format_reward": record["format_reward"],
            "answer_reward": record["answer_reward"],
        }
        for record in records
    }

    def cached_reward(response: str, ground_truth: str) -> dict[str, float]:
        return score_cache[(response, ground_truth)]

    advantages, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=cached_reward,
        rollout_responses=[record["response"] for record in records],
        repeated_ground_truths=[record["ground_truth"] for record in records],
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )
    for record, advantage, raw_reward in zip(records, advantages, raw_rewards):
        record["advantage"] = float(advantage.item())
        record["raw_reward"] = float(raw_reward.item())
    return advantages, raw_rewards, metadata


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def summarize_rollouts(
    records: list[dict[str, Any]],
    advantage_metadata: dict[str, float],
    grpo_step: int,
    group_size: int,
    rollout_seconds: float,
) -> dict[str, Any]:
    lengths = [
        float(record["response_token_length"])
        for record in records
        if record.get("response_token_length") is not None
    ]
    num_questions = len({record["question_index"] for record in records})
    return {
        "type": "rollout",
        "grpo_step": grpo_step,
        "num_questions": num_questions,
        "group_size": group_size,
        "num_rollouts": len(records),
        "reward": mean(record["reward"] for record in records) if records else 0.0,
        "format_accuracy": mean(record["format_reward"] for record in records)
        if records
        else 0.0,
        "answer_accuracy": mean(record["answer_reward"] for record in records)
        if records
        else 0.0,
        "avg_response_token_length": mean_or_none(lengths),
        "rollout_seconds": rollout_seconds,
        **advantage_metadata,
    }


def evaluate_policy(
    policy: torch.nn.Module,
    llm: LLM,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_dir: Path,
    log_dir: Path,
    metrics_path: Path,
    grpo_step: int,
    optimizer_step: int,
) -> dict[str, Any]:
    start = time.time()
    load_policy_into_vllm_instance(policy, llm)
    prompts = [prompt_template.format(question=get_question(example)) for example in val_examples]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)

    records: list[dict[str, Any]] = []
    for example, prompt, output in zip(val_examples, prompts, outputs):
        response = output.outputs[0].text
        ground_truth = get_ground_truth(example)
        scores = r1_zero_reward_fn(response, ground_truth)
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                **scores,
            }
        )

    summary = {
        "num_examples": len(records),
        "reward": mean(record["reward"] for record in records),
        "format_accuracy": mean(record["format_reward"] for record in records),
        "answer_accuracy": mean(record["answer_reward"] for record in records),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stop": "</answer>",
        "include_stop_str_in_output": True,
        "eval_seconds": time.time() - start,
    }
    append_jsonl(
        metrics_path,
        {
            "type": "eval",
            "grpo_step": grpo_step,
            "optimizer_step": optimizer_step,
            **summary,
        },
    )
    generation_path = output_dir / f"eval_generations_grpo_step_{grpo_step:06d}.jsonl"
    for record in records:
        append_jsonl(generation_path, record)
    write_json(log_dir / f"eval_summary_grpo_step_{grpo_step:06d}.json", summary)
    return summary


def iter_chunks(indices: list[int], chunk_size: int):
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


def tokenize_rollout_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, torch.Tensor]:
    return tokenize_prompt_and_output(
        [record["prompt"] for record in records],
        [record["response"] for record in records],
        tokenizer,
    )


def compute_old_log_probs(
    policy: torch.nn.Module,
    rollout_tensors: dict[str, torch.Tensor],
    device: str,
    microbatch_size: int,
) -> torch.Tensor:
    was_training = policy.training
    policy.eval()
    input_ids = rollout_tensors["input_ids"]
    labels = rollout_tensors["labels"]
    old_log_probs: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, input_ids.shape[0], microbatch_size):
            end = start + microbatch_size
            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids[start:end].to(device),
                labels=labels[start:end].to(device),
                return_token_entropy=False,
            )
            old_log_probs.append(scored["log_probs"].detach().cpu())

    if was_training:
        policy.train()
    return torch.cat(old_log_probs, dim=0)


def train_on_rollout_batch(
    policy: torch.nn.Module,
    optimizer: AdamW,
    records: list[dict[str, Any]],
    rollout_tensors: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    epochs_per_rollout_batch: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    loss_type: str,
    cliprange: float,
    device: str,
    rng: random.Random,
    metrics_path: Path,
    grpo_step: int,
    optimizer_step: int,
    learning_rate: float,
    loss_normalization: str,
    loss_normalize_constant: float,
) -> int:
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if len(records) % train_batch_size != 0:
        raise ValueError("rollout batch size must be divisible by train_batch_size.")

    microbatch_size = train_batch_size // gradient_accumulation_steps
    num_train_batches = len(records) // train_batch_size

    policy.train()

    for rollout_epoch in range(1, epochs_per_rollout_batch + 1):
        order = list(range(len(records)))
        rng.shuffle(order)
        train_batches = list(iter_chunks(order, train_batch_size))

        for train_batch_index, train_indices in enumerate(train_batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))

            micro_scaled_losses: list[float] = []
            micro_losses: list[float] = []
            micro_entropies: list[float] = []
            micro_clip_fractions: list[float] = []
            micro_approx_kls: list[float] = []

            for micro_indices in iter_chunks(train_indices, microbatch_size):
                input_ids = rollout_tensors["input_ids"][micro_indices].to(device)
                labels = rollout_tensors["labels"][micro_indices].to(device)
                response_mask = rollout_tensors["response_mask"][micro_indices].to(device)

                scored = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
                micro_advantages = advantages[micro_indices].view(-1, 1).to(device)
                micro_raw_rewards = raw_rewards[micro_indices].view(-1, 1).to(device)
                micro_old_log_probs = (
                    old_log_probs[micro_indices].to(device)
                    if old_log_probs is not None
                    else None
                )
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=scored["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=micro_raw_rewards,
                    advantages=micro_advantages,
                    old_log_probs=micro_old_log_probs,
                    cliprange=cliprange,
                    loss_normalization=loss_normalization,
                    loss_normalize_constant=loss_normalize_constant,
                )
                token_entropy = masked_mean(
                    tensor=scored["token_entropy"],
                    mask=response_mask,
                    dim=None,
                )
                micro_scaled_losses.append(float(loss.detach().cpu().item()))
                micro_losses.append(float(metadata["loss"].detach().cpu().item()))
                micro_entropies.append(float(token_entropy.detach().cpu().item()))

                if micro_old_log_probs is not None:
                    approx_kl = masked_mean(
                        tensor=micro_old_log_probs - scored["log_probs"],
                        mask=response_mask,
                        dim=None,
                    )
                    micro_approx_kls.append(float(approx_kl.detach().cpu().item()))
                if "is_clipped" in metadata:
                    clip_fraction = masked_mean(
                        tensor=metadata["is_clipped"].to(dtype=torch.float32),
                        mask=response_mask,
                        dim=None,
                    )
                    micro_clip_fractions.append(float(clip_fraction.detach().cpu().item()))

                del input_ids, labels, response_mask, scored
                del micro_advantages, micro_raw_rewards, micro_old_log_probs
                del loss, metadata, token_entropy

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_records = [records[index] for index in train_indices]
            optimizer_step += 1
            append_jsonl(
                metrics_path,
                {
                    "type": "train",
                    "grpo_step": grpo_step,
                    "optimizer_step": optimizer_step,
                    "rollout_epoch": rollout_epoch,
                    "epochs_per_rollout_batch": epochs_per_rollout_batch,
                    "train_batch_index": train_batch_index,
                    "num_train_batches": num_train_batches,
                    "loss": mean(micro_losses),
                    "scaled_loss": mean(micro_scaled_losses),
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    "token_entropy": mean(micro_entropies),
                    "clip_fraction": mean_or_none(micro_clip_fractions),
                    "approx_kl": mean_or_none(micro_approx_kls),
                    "train_reward": mean(record["reward"] for record in train_records),
                    "train_format_accuracy": mean(
                        record["format_reward"] for record in train_records
                    ),
                    "train_answer_accuracy": mean(
                        record["answer_reward"] for record in train_records
                    ),
                    "learning_rate": learning_rate,
                    "loss_type": loss_type,
                    "cliprange": cliprange,
                    "loss_normalization": loss_normalization,
                    "loss_normalize_constant": loss_normalize_constant,
                    "rollout_batch_size": len(records),
                    "train_batch_size": train_batch_size,
                    "microbatch_size": microbatch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "old_log_probs_cached": old_log_probs is not None,
                    **cuda_memory_metrics(device),
                },
            )
    return optimizer_step


def validate_grpo_args(args: argparse.Namespace) -> None:
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if args.train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.epochs_per_rollout_batch <= 0:
        raise ValueError("epochs_per_rollout_batch must be positive.")
    if args.train_batch_size > args.rollout_batch_size:
        raise ValueError("train_batch_size must be less than or equal to rollout_batch_size.")
    if args.rollout_batch_size % args.train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by train_batch_size.")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if args.train_batch_size < args.gradient_accumulation_steps:
        raise ValueError("gradient_accumulation_steps must be no larger than train_batch_size.")
    is_off_policy = (
        args.epochs_per_rollout_batch > 1
        or args.train_batch_size != args.rollout_batch_size
    )
    if is_off_policy and args.loss_type not in {"grpo_clip", "grpo_no_clip"}:
        raise ValueError("Off-policy GRPO should use loss_type in {'grpo_clip', 'grpo_no_clip'}.")
    if args.cliprange < 0:
        raise ValueError("cliprange must be non-negative.")
    if args.loss_normalization == "masked_normalize":
        if args.loss_normalize_constant is None:
            args.loss_normalize_constant = float(args.max_new_tokens)
        if args.loss_normalize_constant <= 0:
            raise ValueError("loss_normalize_constant must be positive.")
    elif args.loss_normalize_constant is None:
        args.loss_normalize_constant = 1.0
    if args.eval_every <= 0:
        raise ValueError("eval_every must be positive.")


def expected_optimizer_updates_per_rollout(
    rollout_batch_size: int,
    train_batch_size: int,
    epochs_per_rollout_batch: int,
) -> int:
    return epochs_per_rollout_batch * (rollout_batch_size // train_batch_size)


def main() -> None:
    args = parse_args()
    validate_grpo_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    metrics_path = log_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

    train_examples = read_jsonl(args.train_path, max_examples=args.max_train_questions)
    eval_max_examples = args.eval_max_examples if args.eval_max_examples > 0 else None
    val_examples = read_jsonl(args.val_path, max_examples=eval_max_examples)
    prompt_template = load_prompt_template(args.prompt_template)

    policy, tokenizer = init_policy(args.model, args.policy_device)
    llm = init_vllm(
        model_id=args.model,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    optimizer = AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    rng = random.Random(args.seed)
    optimizer_step = 0
    best_answer_accuracy = -1.0
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    optimizer_updates_per_rollout = expected_optimizer_updates_per_rollout(
        rollout_batch_size=args.rollout_batch_size,
        train_batch_size=args.train_batch_size,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
    )

    logger.info("train question pool: %d", len(train_examples))
    logger.info("validation examples: %d", len(val_examples))
    logger.info("GRPO steps: %d", args.n_grpo_steps)
    logger.info("rollout responses per step: %d", args.rollout_batch_size)
    logger.info("questions per rollout batch: %d", n_prompts_per_rollout_batch)
    logger.info("group size: %d", args.group_size)
    logger.info("epochs per rollout batch: %d", args.epochs_per_rollout_batch)
    logger.info("train batch size: %d", args.train_batch_size)
    logger.info("gradient accumulation steps: %d", args.gradient_accumulation_steps)
    logger.info("optimizer updates per rollout: %d", optimizer_updates_per_rollout)
    logger.info("loss type: %s", args.loss_type)
    logger.info("cliprange: %.4f", args.cliprange)
    logger.info("loss normalization: %s", args.loss_normalization)
    logger.info("loss normalize constant: %.4f", args.loss_normalize_constant)
    logger.info("std normalization: %s", args.use_std_normalization)

    initial_summary = evaluate_policy(
        policy=policy,
        llm=llm,
        val_examples=val_examples,
        prompt_template=prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        output_dir=output_dir,
        log_dir=log_dir,
        metrics_path=metrics_path,
        grpo_step=0,
        optimizer_step=optimizer_step,
    )
    best_answer_accuracy = float(initial_summary["answer_accuracy"])

    for grpo_step in tqdm(range(1, args.n_grpo_steps + 1), desc="GRPO steps"):
        step_start = time.time()
        question_batch = sample_question_batch(
            examples=train_examples,
            num_questions=n_prompts_per_rollout_batch,
            rng=rng,
        )

        load_policy_into_vllm_instance(policy, llm)
        rollout_start = time.time()
        rollout_records = generate_rollouts(
            llm=llm,
            question_batch=question_batch,
            prompt_template=prompt_template,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.sampling_temperature,
            top_p=args.sampling_top_p,
            seed=args.seed + grpo_step,
        )
        rollout_seconds = time.time() - rollout_start
        advantages, raw_rewards, advantage_metadata = attach_advantages(
            records=rollout_records,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        step_output_dir = output_dir / f"grpo_step_{grpo_step:06d}"
        step_log_dir = log_dir / f"grpo_step_{grpo_step:06d}"
        write_jsonl(step_output_dir / "rollouts.jsonl", rollout_records)
        if args.sample_rollouts_to_log > 0:
            write_jsonl(
                step_log_dir / "sample_rollouts.jsonl",
                rollout_records[: args.sample_rollouts_to_log],
            )

        rollout_summary = summarize_rollouts(
            records=rollout_records,
            advantage_metadata=advantage_metadata,
            grpo_step=grpo_step,
            group_size=args.group_size,
            rollout_seconds=rollout_seconds,
        )
        append_jsonl(metrics_path, rollout_summary)
        write_json(step_log_dir / "rollout_summary.json", rollout_summary)

        rollout_tensors = tokenize_rollout_records(rollout_records, tokenizer)
        old_log_probs = None
        if args.loss_type in {"grpo_clip", "grpo_no_clip"}:
            old_log_probs = compute_old_log_probs(
                policy=policy,
                rollout_tensors=rollout_tensors,
                device=args.policy_device,
                microbatch_size=args.train_batch_size // args.gradient_accumulation_steps,
            )

        optimizer_step = train_on_rollout_batch(
            policy=policy,
            optimizer=optimizer,
            records=rollout_records,
            rollout_tensors=rollout_tensors,
            advantages=advantages,
            raw_rewards=raw_rewards,
            old_log_probs=old_log_probs,
            epochs_per_rollout_batch=args.epochs_per_rollout_batch,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            loss_type=args.loss_type,
            cliprange=args.cliprange,
            device=args.policy_device,
            rng=rng,
            metrics_path=metrics_path,
            grpo_step=grpo_step,
            optimizer_step=optimizer_step,
            learning_rate=args.learning_rate,
            loss_normalization=args.loss_normalization,
            loss_normalize_constant=args.loss_normalize_constant,
        )

        if grpo_step % args.eval_every == 0:
            eval_summary = evaluate_policy(
                policy=policy,
                llm=llm,
                val_examples=val_examples,
                prompt_template=prompt_template,
                max_new_tokens=args.max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                output_dir=output_dir,
                log_dir=log_dir,
                metrics_path=metrics_path,
                grpo_step=grpo_step,
                optimizer_step=optimizer_step,
            )
            answer_accuracy = float(eval_summary["answer_accuracy"])
            if answer_accuracy > best_answer_accuracy:
                best_answer_accuracy = answer_accuracy
                if args.save_checkpoints:
                    policy.save_pretrained(output_dir / "best_policy")
                    tokenizer.save_pretrained(output_dir / "best_policy")

        append_jsonl(
            metrics_path,
            {
                "type": "grpo_step_summary",
                "grpo_step": grpo_step,
                "optimizer_step": optimizer_step,
                "step_seconds": time.time() - step_start,
                "best_answer_accuracy": best_answer_accuracy,
                "optimizer_updates_per_rollout_batch": optimizer_updates_per_rollout,
            },
        )

    if args.n_grpo_steps % args.eval_every != 0:
        final_eval = evaluate_policy(
            policy=policy,
            llm=llm,
            val_examples=val_examples,
            prompt_template=prompt_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            output_dir=output_dir,
            log_dir=log_dir,
            metrics_path=metrics_path,
            grpo_step=args.n_grpo_steps,
            optimizer_step=optimizer_step,
        )
        final_answer_accuracy = float(final_eval["answer_accuracy"])
        if final_answer_accuracy > best_answer_accuracy:
            best_answer_accuracy = final_answer_accuracy
            if args.save_checkpoints:
                policy.save_pretrained(output_dir / "best_policy")
                tokenizer.save_pretrained(output_dir / "best_policy")

    run_summary = {
        "best_answer_accuracy": best_answer_accuracy,
        "final_optimizer_step": optimizer_step,
        "n_grpo_steps": args.n_grpo_steps,
        "rollout_batch_size": args.rollout_batch_size,
        "group_size": args.group_size,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer_updates_per_rollout_batch": optimizer_updates_per_rollout,
        "expected_total_optimizer_steps": args.n_grpo_steps * optimizer_updates_per_rollout,
        "loss_type": args.loss_type,
        "cliprange": args.cliprange,
        "loss_normalization": args.loss_normalization,
        "loss_normalize_constant": args.loss_normalize_constant,
        "use_std_normalization": args.use_std_normalization,
    }
    write_json(log_dir / "run_summary.json", run_summary)
    if args.save_checkpoints:
        policy.save_pretrained(output_dir / "final_policy")
        tokenizer.save_pretrained(output_dir / "final_policy")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
