#!/usr/bin/env python3
"""Isolate the first immediate sleep/wake sequence on manager-owned first inference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from vllm import SamplingParams

from cs336_alignment.backend_lifecycle import (
    BackendLifecycleManager,
    init_vllm,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_VAL_PATH = (
    "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl"
)
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR_BASE = ".agents/logs/diagnose_first_sleep_wake_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current manager first-inference path against variants that "
            "skip the initial sleep/wake or disable sleep mode entirely."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("current_manager", "skip_first_sleep_wake", "sleep_disabled_manager"),
        required=True,
        help="Which first-inference path to run in this process.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional override for the output directory. Defaults to a mode-specific path.",
    )
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-level", type=int, default=2)
    parser.add_argument("--sample-dump-count", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_examples(
    examples: list[dict[str, Any]],
    *,
    num_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_examples >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [examples[index] for index in indices[:num_examples]]


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


def build_manager(args: argparse.Namespace, *, enable_sleep_mode: bool) -> BackendLifecycleManager:
    return BackendLifecycleManager.from_defaults(
        model_id=args.model,
        policy_device=args.policy_device,
        vllm_device=args.vllm_device,
        seed=args.seed,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_sleep_mode=enable_sleep_mode,
        sleep_level=args.sleep_level,
        keep_policy_resident_on_device=True,
        keep_optimizer_state_resident_on_device=False,
    )


def score_outputs(
    *,
    examples: list[dict[str, Any]],
    prompts: list[str],
    outputs: list[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example, prompt, output in zip(examples, prompts, outputs):
        response = output.outputs[0].text
        scores = r1_zero_reward_fn(response, get_ground_truth(example))
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": get_ground_truth(example),
                **scores,
            }
        )
    return records


def summarize_records(
    records: list[dict[str, Any]],
    *,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "num_examples": len(records),
        "reward": mean(record["reward"] for record in records),
        "format_accuracy": mean(record["format_reward"] for record in records),
        "answer_accuracy": mean(record["answer_reward"] for record in records),
        **extra,
    }


def run_manager_current(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompts: list[str],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    manager = build_manager(args, enable_sleep_mode=True)
    manager.enter_training_phase()
    state_before_eval = manager.debug_state()
    llm = manager.enter_inference_phase(sync_weights=False)
    state_during_inference = manager.debug_state()
    outputs = llm.generate(prompts, sampling_params)
    manager.enter_training_phase()
    state_after_eval = manager.debug_state()
    records = score_outputs(examples=examples, prompts=prompts, outputs=outputs)
    return {
        "summary": summarize_records(
            records,
            extra={
                "mode": "current_manager",
                "state_before_eval": state_before_eval,
                "state_during_inference": state_during_inference,
                "state_after_eval": state_after_eval,
            },
        ),
        "records": records,
    }


def run_skip_first_sleep_wake(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompts: list[str],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    manager = build_manager(args, enable_sleep_mode=True)
    manager.enter_training_phase()

    llm = init_vllm(
        model_id=args.model,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_sleep_mode=True,
    )
    manager._llm = llm
    manager._inference_backend_awake = True

    state_before_eval = manager.debug_state()
    llm = manager.enter_inference_phase(sync_weights=False)
    state_during_inference = manager.debug_state()
    outputs = llm.generate(prompts, sampling_params)
    manager.enter_training_phase()
    state_after_eval = manager.debug_state()
    records = score_outputs(examples=examples, prompts=prompts, outputs=outputs)
    return {
        "summary": summarize_records(
            records,
            extra={
                "mode": "skip_first_sleep_wake",
                "state_before_eval": state_before_eval,
                "state_during_inference": state_during_inference,
                "state_after_eval": state_after_eval,
            },
        ),
        "records": records,
    }


def run_sleep_disabled_manager(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompts: list[str],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    manager = build_manager(args, enable_sleep_mode=False)
    manager.enter_training_phase()
    state_before_eval = manager.debug_state()
    llm = manager.enter_inference_phase(sync_weights=False)
    state_during_inference = manager.debug_state()
    outputs = llm.generate(prompts, sampling_params)
    manager.enter_training_phase()
    state_after_eval = manager.debug_state()
    records = score_outputs(examples=examples, prompts=prompts, outputs=outputs)
    return {
        "summary": summarize_records(
            records,
            extra={
                "mode": "sleep_disabled_manager",
                "state_before_eval": state_before_eval,
                "state_during_inference": state_during_inference,
                "state_after_eval": state_after_eval,
            },
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir or f"{DEFAULT_LOG_DIR_BASE}_{args.mode}")
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

    all_examples = read_jsonl(args.val_path)
    eval_examples = select_examples(all_examples, num_examples=args.num_examples, seed=args.seed)
    prompt_template = load_prompt_template(args.prompt_template)
    prompts = [prompt_template.format(question=get_question(example)) for example in eval_examples]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    if args.mode == "current_manager":
        result = run_manager_current(
            args,
            examples=eval_examples,
            prompts=prompts,
            sampling_params=sampling_params,
        )
    elif args.mode == "skip_first_sleep_wake":
        result = run_skip_first_sleep_wake(
            args,
            examples=eval_examples,
            prompts=prompts,
            sampling_params=sampling_params,
        )
    else:
        result = run_sleep_disabled_manager(
            args,
            examples=eval_examples,
            prompts=prompts,
            sampling_params=sampling_params,
        )

    write_json(log_dir / "summary.json", result["summary"])
    write_jsonl(log_dir / "records.jsonl", result["records"])
    write_jsonl(log_dir / "sample_records.jsonl", result["records"][: args.sample_dump_count])


if __name__ == "__main__":
    main()
