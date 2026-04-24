#!/usr/bin/env python3
"""Run one MATH SFT experiment with periodic vLLM validation."""

from __future__ import annotations

import argparse
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

from cs336_alignment.backend_lifecycle import BackendLifecycleManager
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.experiment_logging import (
    ExperimentLogWriter,
    get_ground_truth,
    get_question,
    load_prompt_template,
    read_jsonl,
)
from cs336_alignment.experiment_metrics import cuda_memory_metrics
from cs336_alignment.reward_scoring import score_responses
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/sft.jsonl"
DEFAULT_VAL_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl"
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = ".agents/logs/ch4/sft_experiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT on MATH reasoning traces.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for large outputs such as checkpoints and full generation dumps.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for small logs, metrics, configs, and summaries.",
    )
    parser.add_argument("--num-train-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=None,
        help="Maximum number of validation examples to evaluate. Defaults to the full set.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Effective number of SFT examples per optimizer step.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--normalize-constant",
        type=float,
        default=1.0,
        help="Constant used inside masked_normalize for each example.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument(
        "--reward-workers",
        type=int,
        default=1,
        help="Number of worker processes for CPU-side validation reward scoring.",
    )
    parser.add_argument(
        "--quiet-reward-parser-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress noisy dependency parser logs during CPU-side reward scoring.",
    )
    parser.add_argument(
        "--reward-timeout-seconds",
        type=int,
        default=5,
        help="Per-response reward scoring timeout. Timed-out responses receive zero reward.",
    )
    parser.add_argument(
        "--filter-correct-sft",
        action="store_true",
        help="Keep only SFT traces that receive answer_reward == 1.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_policy and final_policy under --output-dir. Leave off for smoke/debug runs.",
    )
    return parser.parse_args()


def maybe_filter_correct_sft(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for example in examples:
        scores = r1_zero_reward_fn(example["response"], get_ground_truth(example))
        if scores["answer_reward"] == 1.0:
            filtered.append(example)
    return filtered


def sample_train_examples(
    examples: list[dict[str, Any]],
    num_train_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if num_train_examples is None or num_train_examples >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(examples, num_train_examples)


def iter_minibatches(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[i] for i in order[start : start + batch_size]]


def evaluate_with_vllm(
    llm: LLM,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward_workers: int,
    quiet_reward_parser_logs: bool,
    reward_timeout_seconds: int | None,
) -> dict[str, Any]:
    prompts = [prompt_template.format(question=get_question(example)) for example in val_examples]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)

    scoring_items = [
        (prompt, output.outputs[0].text, get_ground_truth(example))
        for example, prompt, output in zip(val_examples, prompts, outputs)
    ]
    responses = [response for _, response, _ in scoring_items]
    ground_truths = [ground_truth for _, _, ground_truth in scoring_items]
    scores_list = score_responses(
        responses=responses,
        ground_truths=ground_truths,
        reward_fn_name="r1_zero",
        reward_workers=reward_workers,
        quiet_parser_logs=quiet_reward_parser_logs,
        timeout_seconds=reward_timeout_seconds,
    )

    records: list[dict[str, Any]] = []
    for (prompt, response, ground_truth), scores in zip(scoring_items, scores_list):
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                **scores,
            }
        )

    return {
        "records": records,
        "summary": {
            "num_examples": len(records),
            "reward": mean(r["reward"] for r in records),
            "format_accuracy": mean(r["format_reward"] for r in records),
            "answer_accuracy": mean(r["answer_reward"] for r in records),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "reward_workers": reward_workers,
            "quiet_reward_parser_logs": quiet_reward_parser_logs,
            "reward_timeout_seconds": reward_timeout_seconds,
            "stop": "</answer>",
            "include_stop_str_in_output": True,
        },
    }


def run_eval_and_log(
    backend_manager: BackendLifecycleManager,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    reward_workers: int,
    quiet_reward_parser_logs: bool,
    reward_timeout_seconds: int | None,
    writer: ExperimentLogWriter,
    train_step: int,
) -> dict[str, Any]:
    start = time.time()
    llm = backend_manager.enter_inference_phase(sync_weights=True)
    eval_result = evaluate_with_vllm(
        llm=llm,
        val_examples=val_examples,
        prompt_template=prompt_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        reward_workers=reward_workers,
        quiet_reward_parser_logs=quiet_reward_parser_logs,
        reward_timeout_seconds=reward_timeout_seconds,
    )
    summary = eval_result["summary"]
    writer.append_metric(
        {
            "type": "eval",
            "train_step": train_step,
            "eval_seconds": time.time() - start,
            **summary,
        },
    )
    for record in eval_result["records"]:
        writer.append_eval_generation(train_step, record)
    writer.write_eval_summary(train_step, summary)
    backend_manager.enter_training_phase()
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    writer = ExperimentLogWriter("sft", output_dir=output_dir, log_dir=log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer.write_config(vars(args))

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    microbatch_size = args.train_batch_size // args.gradient_accumulation_steps

    train_examples = read_jsonl(args.train_path)
    if args.filter_correct_sft:
        train_examples = maybe_filter_correct_sft(train_examples)
        logger.info("filtered SFT dataset size: %d", len(train_examples))
    train_examples = sample_train_examples(
        train_examples,
        num_train_examples=args.num_train_examples,
        seed=args.seed,
    )
    val_examples = read_jsonl(args.val_path, max_examples=args.eval_max_examples)
    prompt_template = load_prompt_template(args.prompt_template)

    backend_manager = BackendLifecycleManager.from_defaults(
        model_id=args.model,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        seed=args.seed,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_sleep_mode=True,
        sleep_level=2,
        keep_policy_resident_on_device=True,
        keep_optimizer_state_resident_on_device=False,
    )
    policy, tokenizer, optimizer = backend_manager.initialize_rl_runtime(
        optimizer_factory=lambda parameters: AdamW(parameters, lr=args.learning_rate)
    )
    policy, tokenizer = backend_manager.enter_training_phase()
    rng = random.Random(args.seed)
    optimizer.zero_grad(set_to_none=True)

    logger.info("training examples: %d", len(train_examples))
    logger.info("validation examples: %d", len(val_examples))
    logger.info("policy device: %s; vLLM device: %s", args.policy_device, args.vllm_device)
    logger.info("effective train batch size: %d", args.train_batch_size)
    logger.info("microbatch size: %d", microbatch_size)

    best_answer_accuracy = -1.0
    train_iter = iter_minibatches(train_examples, microbatch_size, rng)
    for train_step in tqdm(range(args.max_steps), desc="SFT optimizer steps"):
        if args.policy_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(args.policy_device))
        micro_scaled_losses: list[float] = []
        micro_losses: list[float] = []

        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter_minibatches(train_examples, microbatch_size, rng)
                batch = next(train_iter)

            prompts = [example["prompt"] for example in batch]
            responses = [example["response"] for example in batch]
            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(args.policy_device)
            labels = tokenized["labels"].to(args.policy_device)
            response_mask = tokenized["response_mask"].to(args.policy_device)

            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=args.normalize_constant,
            )
            micro_scaled_losses.append(float(loss.detach().cpu().item()))
            micro_losses.append(float(metadata["loss"].detach().cpu().item()))
            del scored, loss, metadata
            del input_ids, labels, response_mask, tokenized

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        writer.append_metric(
            {
                "type": "train",
                "train_step": train_step,
                "loss": mean(micro_losses),
                "scaled_loss": mean(micro_scaled_losses),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "num_train_examples": len(train_examples),
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "microbatch_size": microbatch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "normalize_constant": args.normalize_constant,
                **cuda_memory_metrics(args.policy_device),
            },
        )

        if train_step % args.eval_every == 0:
            summary = run_eval_and_log(
                backend_manager=backend_manager,
                val_examples=val_examples,
                prompt_template=prompt_template,
                max_new_tokens=args.max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                reward_workers=args.reward_workers,
                quiet_reward_parser_logs=args.quiet_reward_parser_logs,
                reward_timeout_seconds=args.reward_timeout_seconds,
                writer=writer,
                train_step=train_step,
            )
            answer_accuracy = float(summary["answer_accuracy"])
            if answer_accuracy > best_answer_accuracy:
                best_answer_accuracy = answer_accuracy
                if args.save_checkpoints:
                    policy.save_pretrained(output_dir / "best_policy")
                    tokenizer.save_pretrained(output_dir / "best_policy")

    final_step = args.max_steps
    summary = run_eval_and_log(
        backend_manager=backend_manager,
        val_examples=val_examples,
        prompt_template=prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        reward_workers=args.reward_workers,
        quiet_reward_parser_logs=args.quiet_reward_parser_logs,
        reward_timeout_seconds=args.reward_timeout_seconds,
        writer=writer,
        train_step=final_step,
    )
    if float(summary["answer_accuracy"]) > best_answer_accuracy and args.save_checkpoints:
        policy.save_pretrained(output_dir / "best_policy")
        tokenizer.save_pretrained(output_dir / "best_policy")

    if args.save_checkpoints:
        policy.save_pretrained(output_dir / "final_policy")
        tokenizer.save_pretrained(output_dir / "final_policy")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
