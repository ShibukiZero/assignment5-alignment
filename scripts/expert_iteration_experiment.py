#!/usr/bin/env python3
"""Run Expert Iteration on MATH-style reasoning tasks."""

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
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from cs336_alignment.backend_lifecycle import BackendLifecycleManager
from cs336_alignment.experiment_logging import (
    ExperimentLogWriter,
    get_ground_truth,
    get_question,
    load_prompt_template,
    read_jsonl,
)
from cs336_alignment.experiment_metrics import cuda_memory_metrics, mean_or_none
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.reward_scoring import quiet_reward_parser_logs, score_response_with_reward_fn
from cs336_alignment.sft import (
    get_response_log_probs,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

from sft_experiment import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
)


logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/train.jsonl"
DEFAULT_VAL_PATH = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric/validation.jsonl"
DEFAULT_LOG_DIR = ".agents/logs/ch5/expert_iteration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Expert Iteration on MATH-style tasks.")
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
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=512,
        help="Number of unique questions sampled at each EI step.",
    )
    parser.add_argument(
        "--rollouts-per-question",
        type=int,
        default=4,
        help="Number of sampled responses per question.",
    )
    parser.add_argument(
        "--sft-epochs-per-step",
        type=float,
        default=1.0,
        help="Number of SFT epochs over the accepted traces at each EI step.",
    )
    parser.add_argument(
        "--max-train-questions",
        type=int,
        default=None,
        help="Optional cap on the train question pool for smoke runs.",
    )
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=None,
        help="Maximum number of validation examples to evaluate. Defaults to the full set.",
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--normalize-constant", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--reset-optimizer-each-ei-step",
        action="store_true",
        help="Reset AdamW state before training on each accepted EI dataset.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument(
        "--filter-reward-key",
        default="answer_reward",
        choices=["reward", "answer_reward", "format_reward"],
        help="Reward field used to decide whether a rollout becomes an SFT trace.",
    )
    parser.add_argument("--filter-threshold", type=float, default=1.0)
    parser.add_argument("--entropy-batch-size", type=int, default=4)
    parser.add_argument(
        "--entropy-max-records",
        type=int,
        default=None,
        help="Optionally score entropy on only a sampled subset of rollout records.",
    )
    parser.add_argument(
        "--no-score-rollout-entropy",
        dest="score_rollout_entropy",
        action="store_false",
        help="Skip HF entropy scoring for rollout responses.",
    )
    parser.set_defaults(score_rollout_entropy=True)
    parser.add_argument(
        "--quiet-reward-parser-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress noisy dependency parser logs during reward scoring.",
    )
    parser.add_argument(
        "--reward-timeout-seconds",
        type=int,
        default=5,
        help="Per-response reward scoring timeout. Timed-out responses receive zero reward.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best_policy and final_policy under --output-dir.",
    )
    return parser.parse_args()


def sample_question_batch(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(examples))
    if batch_size >= len(indexed):
        rng.shuffle(indexed)
        return indexed
    return rng.sample(indexed, batch_size)


def build_rollout_records(
    question_batch: list[tuple[int, dict[str, Any]]],
    prompt_template: str,
    outputs: list[Any],
    reward_timeout_seconds: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (question_index, example), request_output in zip(question_batch, outputs):
        question = get_question(example)
        prompt = prompt_template.format(question=question)
        ground_truth = get_ground_truth(example)
        for rollout_index, completion in enumerate(request_output.outputs):
            response = completion.text
            scores = score_response_with_reward_fn(
                response=response,
                ground_truth=ground_truth,
                reward_fn=r1_zero_reward_fn,
                timeout_seconds=reward_timeout_seconds,
            )
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
    rollouts_per_question: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    reward_timeout_seconds: int | None,
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
        n=rollouts_per_question,
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    return build_rollout_records(
        question_batch=question_batch,
        prompt_template=prompt_template,
        outputs=outputs,
        reward_timeout_seconds=reward_timeout_seconds,
    )


def score_response_entropy(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    records: list[dict[str, Any]],
    batch_size: int,
    device: str,
) -> list[float]:
    if not records:
        return []

    was_training = policy.training
    policy.eval()
    entropies: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            tokenized = tokenize_prompt_and_output(
                [record["prompt"] for record in batch],
                [record["response"] for record in batch],
                tokenizer,
            )
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)
            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            per_example_entropy = masked_normalize(
                tensor=scored["token_entropy"],
                mask=response_mask,
                dim=-1,
                normalize_constant=1.0,
            ) / response_mask.sum(dim=-1).clamp(min=1)
            entropies.extend(
                float(value)
                for value in per_example_entropy.detach().cpu().tolist()
            )
            del input_ids, labels, response_mask, tokenized, scored, per_example_entropy

    if was_training:
        policy.train()
    return entropies


def annotate_entropy(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    records: list[dict[str, Any]],
    batch_size: int,
    device: str,
    max_records: int | None,
    rng: random.Random,
) -> None:
    if max_records is not None and max_records < len(records):
        records_to_score = rng.sample(records, max_records)
    else:
        records_to_score = records
    entropies = score_response_entropy(
        policy=policy,
        tokenizer=tokenizer,
        records=records_to_score,
        batch_size=batch_size,
        device=device,
    )
    for record, entropy in zip(records_to_score, entropies):
        record["avg_token_entropy"] = entropy


def accepted_sft_records(
    records: list[dict[str, Any]],
    reward_key: str,
    threshold: float,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for record in records:
        if float(record[reward_key]) >= threshold:
            accepted.append(
                {
                    "prompt": record["prompt"],
                    "response": record["response"],
                    "question": record["question"],
                    "ground_truth": record["ground_truth"],
                    "source_question_index": record["question_index"],
                    "source_rollout_index": record["rollout_index"],
                    "source_reward": record["reward"],
                    "source_format_reward": record["format_reward"],
                    "source_answer_reward": record["answer_reward"],
                    "source_avg_token_entropy": record.get("avg_token_entropy"),
                    "source_response_token_length": record.get("response_token_length"),
                }
            )
    return accepted


def summarize_rollouts(
    records: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    ei_step: int,
    rollouts_per_question: int,
    rollout_seconds: float,
) -> dict[str, Any]:
    entropies = [
        float(record["avg_token_entropy"])
        for record in records
        if record.get("avg_token_entropy") is not None
    ]
    accepted_entropies = [
        float(record["source_avg_token_entropy"])
        for record in accepted
        if record.get("source_avg_token_entropy") is not None
    ]
    lengths = [
        float(record["response_token_length"])
        for record in records
        if record.get("response_token_length") is not None
    ]
    accepted_lengths = [
        float(record["source_response_token_length"])
        for record in accepted
        if record.get("source_response_token_length") is not None
    ]
    num_questions = len({record["question_index"] for record in records})
    return {
        "type": "rollout",
        "ei_step": ei_step,
        "num_questions": num_questions,
        "rollouts_per_question": rollouts_per_question,
        "num_rollouts": len(records),
        "num_accepted": len(accepted),
        "accepted_fraction": len(accepted) / len(records) if records else 0.0,
        "question_coverage_with_accept": len(
            {record["source_question_index"] for record in accepted}
        )
        / num_questions
        if num_questions
        else 0.0,
        "reward": mean(record["reward"] for record in records) if records else 0.0,
        "format_accuracy": mean(record["format_reward"] for record in records)
        if records
        else 0.0,
        "answer_accuracy": mean(record["answer_reward"] for record in records)
        if records
        else 0.0,
        "avg_token_entropy": mean_or_none(entropies),
        "avg_accepted_token_entropy": mean_or_none(accepted_entropies),
        "avg_response_token_length": mean_or_none(lengths),
        "avg_accepted_response_token_length": mean_or_none(accepted_lengths),
        "rollout_seconds": rollout_seconds,
    }


def iter_effective_batches(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def train_on_accepted_traces(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: AdamW,
    accepted: list[dict[str, Any]],
    train_batch_size: int,
    gradient_accumulation_steps: int,
    normalize_constant: float,
    sft_epochs: float,
    device: str,
    rng: random.Random,
    writer: ExperimentLogWriter,
    ei_step: int,
    optimizer_step_start: int,
) -> int:
    if not accepted or sft_epochs <= 0:
        return optimizer_step_start
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")

    microbatch_size = train_batch_size // gradient_accumulation_steps
    full_epochs = int(sft_epochs)
    fractional_epoch = sft_epochs - full_epochs
    epoch_sizes = [len(accepted)] * full_epochs
    if fractional_epoch > 0:
        epoch_sizes.append(max(1, int(round(len(accepted) * fractional_epoch))))

    optimizer_step = optimizer_step_start
    policy.train()
    for epoch_index, epoch_size in enumerate(epoch_sizes):
        if epoch_size >= len(accepted):
            epoch_examples = list(accepted)
        else:
            epoch_examples = rng.sample(accepted, epoch_size)

        progress = tqdm(
            iter_effective_batches(epoch_examples, train_batch_size, rng),
            desc=f"EI step {ei_step} SFT epoch {epoch_index + 1}",
            leave=False,
        )
        for effective_batch in progress:
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            micro_scaled_losses: list[float] = []
            micro_losses: list[float] = []

            microbatches = [
                effective_batch[start : start + microbatch_size]
                for start in range(0, len(effective_batch), microbatch_size)
            ]
            optimizer.zero_grad(set_to_none=True)
            for microbatch in microbatches:
                tokenized = tokenize_prompt_and_output(
                    [example["prompt"] for example in microbatch],
                    [example["response"] for example in microbatch],
                    tokenizer,
                )
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
                    gradient_accumulation_steps=len(microbatches),
                    normalize_constant=normalize_constant,
                )
                micro_scaled_losses.append(float(loss.detach().cpu().item()))
                micro_losses.append(float(metadata["loss"].detach().cpu().item()))
                del input_ids, labels, response_mask, tokenized, scored, loss, metadata

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_step += 1
            writer.append_metric(
                {
                    "type": "train",
                    "ei_step": ei_step,
                    "optimizer_step": optimizer_step,
                    "sft_epoch": epoch_index + 1,
                    "loss": mean(micro_losses),
                    "scaled_loss": mean(micro_scaled_losses),
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    "num_train_examples": len(accepted),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_batch_size": train_batch_size,
                    "microbatch_size": microbatch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "normalize_constant": normalize_constant,
                    **cuda_memory_metrics(device),
                },
            )

    optimizer.zero_grad(set_to_none=True)
    return optimizer_step


def evaluate_policy(
    backend_manager: BackendLifecycleManager,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    writer: ExperimentLogWriter,
    ei_step: int,
    optimizer_step: int,
    reward_timeout_seconds: int | None,
) -> dict[str, Any]:
    start = time.time()
    llm = backend_manager.enter_inference_phase(sync_weights=True)
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
        scores = score_response_with_reward_fn(
            response=response,
            ground_truth=ground_truth,
            reward_fn=r1_zero_reward_fn,
            timeout_seconds=reward_timeout_seconds,
        )
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
    writer.append_metric(
        {
            "type": "eval",
            "ei_step": ei_step,
            "optimizer_step": optimizer_step,
            **summary,
        },
    )
    for record in records:
        writer.append_eval_generation(ei_step, record)
    writer.write_eval_summary(ei_step, summary)
    backend_manager.enter_training_phase()
    return summary


def main() -> None:
    args = parse_args()
    if args.quiet_reward_parser_logs:
        quiet_reward_parser_logs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    writer = ExperimentLogWriter("ei", output_dir=output_dir, log_dir=log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer.write_config(vars(args))

    train_examples = read_jsonl(args.train_path, max_examples=args.max_train_questions)
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
    optimizer_step = 0
    best_answer_accuracy = -1.0

    logger.info("train question pool: %d", len(train_examples))
    logger.info("validation examples: %d", len(val_examples))
    logger.info("EI steps: %d", args.n_ei_steps)
    logger.info("rollout batch size: %d", args.rollout_batch_size)
    logger.info("rollouts per question: %d", args.rollouts_per_question)

    initial_summary = evaluate_policy(
        backend_manager=backend_manager,
        val_examples=val_examples,
        prompt_template=prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        writer=writer,
        ei_step=0,
        optimizer_step=optimizer_step,
        reward_timeout_seconds=args.reward_timeout_seconds,
    )
    best_answer_accuracy = float(initial_summary["answer_accuracy"])

    for ei_step in range(1, args.n_ei_steps + 1):
        step_start = time.time()
        if args.reset_optimizer_each_ei_step:
            optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
            backend_manager.attach_training_optimizer(optimizer)

        question_batch = sample_question_batch(
            examples=train_examples,
            batch_size=args.rollout_batch_size,
            rng=rng,
        )

        llm = backend_manager.enter_inference_phase(sync_weights=True)
        rollout_start = time.time()
        rollout_records = generate_rollouts(
            llm=llm,
            question_batch=question_batch,
            prompt_template=prompt_template,
            rollouts_per_question=args.rollouts_per_question,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            seed=args.seed + ei_step,
            reward_timeout_seconds=args.reward_timeout_seconds,
        )
        rollout_seconds = time.time() - rollout_start

        policy, tokenizer = backend_manager.enter_training_phase()
        if args.score_rollout_entropy:
            annotate_entropy(
                policy=policy,
                tokenizer=tokenizer,
                records=rollout_records,
                batch_size=args.entropy_batch_size,
                device=args.policy_device,
                max_records=args.entropy_max_records,
                rng=rng,
            )

        accepted = accepted_sft_records(
            records=rollout_records,
            reward_key=args.filter_reward_key,
            threshold=args.filter_threshold,
        )
        writer.write_rollouts(ei_step, rollout_records)
        writer.write_accepted_sft(ei_step, accepted)

        rollout_summary = summarize_rollouts(
            records=rollout_records,
            accepted=accepted,
            ei_step=ei_step,
            rollouts_per_question=args.rollouts_per_question,
            rollout_seconds=rollout_seconds,
        )
        writer.append_metric(rollout_summary)
        writer.write_rollout_summary(ei_step, rollout_summary)

        logger.info(
            "EI step %d accepted %d/%d rollouts",
            ei_step,
            len(accepted),
            len(rollout_records),
        )

        optimizer_step = train_on_accepted_traces(
            policy=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accepted=accepted,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            normalize_constant=args.normalize_constant,
            sft_epochs=args.sft_epochs_per_step,
            device=args.policy_device,
            rng=rng,
            writer=writer,
            ei_step=ei_step,
            optimizer_step_start=optimizer_step,
        )

        eval_summary = evaluate_policy(
            backend_manager=backend_manager,
            val_examples=val_examples,
            prompt_template=prompt_template,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            writer=writer,
            ei_step=ei_step,
            optimizer_step=optimizer_step,
            reward_timeout_seconds=args.reward_timeout_seconds,
        )

        answer_accuracy = float(eval_summary["answer_accuracy"])
        if answer_accuracy > best_answer_accuracy:
            best_answer_accuracy = answer_accuracy
            if args.save_checkpoints:
                policy.save_pretrained(output_dir / "best_policy")
                tokenizer.save_pretrained(output_dir / "best_policy")

        writer.append_metric(
            {
                "type": "ei_step_summary",
                "ei_step": ei_step,
                "optimizer_step": optimizer_step,
                "step_seconds": time.time() - step_start,
                "accepted_examples": len(accepted),
                "eval_answer_accuracy": answer_accuracy,
                "best_answer_accuracy": best_answer_accuracy,
            },
        )

    final_summary = {
        "best_answer_accuracy": best_answer_accuracy,
        "final_optimizer_step": optimizer_step,
        "n_ei_steps": args.n_ei_steps,
    }
    writer.write_run_summary(final_summary)
    if args.save_checkpoints:
        policy.save_pretrained(output_dir / "final_policy")
        tokenizer.save_pretrained(output_dir / "final_policy")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
