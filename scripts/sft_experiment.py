#!/usr/bin/env python3
"""Run one MATH SFT experiment with periodic vLLM validation."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any
from unittest.mock import patch

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
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
BYTES_PER_GIB = 1024**3


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


def read_jsonl(path: str, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if max_examples is not None and len(rows) >= max_examples:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


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


def init_policy(model_id: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(device)
    model.train()
    return model, tokenizer


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float,
) -> LLM:
    """Start a vLLM model on a GPU separate from the training policy."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.float16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Copy the current HF policy weights into the already-running vLLM model."""
    state_dict = policy.state_dict()
    llm_engine = getattr(llm, "llm_engine", getattr(llm, "engine", None))
    if llm_engine is None:
        raise AttributeError("Could not find vLLM engine on the LLM object.")
    vllm_model = llm_engine.model_executor.driver_worker.model_runner.model
    vllm_model.load_weights(state_dict.items())


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
            "stop": "</answer>",
            "include_stop_str_in_output": True,
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cuda_memory_metrics(device: str) -> dict[str, float]:
    if not device.startswith("cuda"):
        return {}
    cuda_device = torch.device(device)
    return {
        "cuda_memory_allocated_gib": torch.cuda.memory_allocated(cuda_device) / BYTES_PER_GIB,
        "cuda_memory_reserved_gib": torch.cuda.memory_reserved(cuda_device) / BYTES_PER_GIB,
        "cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated(cuda_device)
        / BYTES_PER_GIB,
        "cuda_max_memory_reserved_gib": torch.cuda.max_memory_reserved(cuda_device)
        / BYTES_PER_GIB,
    }


def run_eval_and_log(
    policy: PreTrainedModel,
    llm: LLM,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_dir: Path,
    log_dir: Path,
    metrics_path: Path,
    train_step: int,
) -> dict[str, Any]:
    start = time.time()
    load_policy_into_vllm_instance(policy, llm)
    eval_result = evaluate_with_vllm(
        llm=llm,
        val_examples=val_examples,
        prompt_template=prompt_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    summary = eval_result["summary"]
    append_jsonl(
        metrics_path,
        {
            "type": "eval",
            "train_step": train_step,
            "eval_seconds": time.time() - start,
            **summary,
        },
    )
    generations_path = output_dir / f"eval_generations_step_{train_step:06d}.jsonl"
    for record in eval_result["records"]:
        append_jsonl(generations_path, record)
    write_json(log_dir / f"eval_summary_step_{train_step:06d}.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    metrics_path = log_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

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

    policy, tokenizer = init_policy(args.model, args.policy_device)
    llm = init_vllm(
        model_id=args.model,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
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

        append_jsonl(
            metrics_path,
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
