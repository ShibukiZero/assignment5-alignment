#!/usr/bin/env python3
"""Compare fresh vLLM eval against the current unified lifecycle path."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

from torch.optim import AdamW
from vllm import SamplingParams

from cs336_alignment.backend_lifecycle import BackendLifecycleManager, init_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_VAL_PATH = (
    "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl"
)
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = ".agents/logs/diagnose_sft_eval_paths"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the unified lifecycle path changes SFT eval behavior."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--vllm-device", default="cuda:0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-level", type=int, default=2)
    parser.add_argument(
        "--sample-dump-count",
        type=int,
        default=5,
        help="How many example generations to dump for each path.",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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


def evaluate_path(
    *,
    llm: Any,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    prompts = [prompt_template.format(question=get_question(example)) for example in examples]
    started_at = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - started_at

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

    summary = {
        "num_examples": len(records),
        "eval_seconds": elapsed,
        "reward": mean(record["reward"] for record in records),
        "format_accuracy": mean(record["format_reward"] for record in records),
        "answer_accuracy": mean(record["answer_reward"] for record in records),
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
        "stop": "</answer>",
        "include_stop_str_in_output": True,
    }
    return {"summary": summary, "records": records}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_fresh_vllm_path(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    llm = init_vllm(
        model_id=args.model,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_sleep_mode=False,
    )
    return evaluate_path(
        llm=llm,
        examples=examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )


def run_lifecycle_sleep_path(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
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
    manager.initialize_rl_runtime(
        optimizer_factory=lambda parameters: AdamW(parameters, lr=args.learning_rate)
    )
    manager.enter_training_phase()
    llm = manager.enter_inference_phase(sync_weights=True)
    result = evaluate_path(
        llm=llm,
        examples=examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )
    manager.enter_training_phase()
    return result


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "config.json", vars(args))

    all_examples = read_jsonl(args.val_path)
    eval_examples = select_examples(
        all_examples,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    prompt_template = load_prompt_template(args.prompt_template)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    fresh_result = run_fresh_vllm_path(
        args,
        examples=eval_examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )
    lifecycle_result = run_lifecycle_sleep_path(
        args,
        examples=eval_examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )

    write_json(log_dir / "fresh_summary.json", fresh_result["summary"])
    write_json(log_dir / "lifecycle_sleep_summary.json", lifecycle_result["summary"])
    write_jsonl(log_dir / "fresh_records.jsonl", fresh_result["records"])
    write_jsonl(log_dir / "lifecycle_sleep_records.jsonl", lifecycle_result["records"])
    write_jsonl(
        log_dir / "fresh_sample_records.jsonl",
        fresh_result["records"][: args.sample_dump_count],
    )
    write_jsonl(
        log_dir / "lifecycle_sleep_sample_records.jsonl",
        lifecycle_result["records"][: args.sample_dump_count],
    )
    write_json(
        log_dir / "comparison_summary.json",
        {
            "fresh": fresh_result["summary"],
            "lifecycle_sleep": lifecycle_result["summary"],
            "delta": {
                key: lifecycle_result["summary"][key] - fresh_result["summary"][key]
                for key in ("reward", "format_accuracy", "answer_accuracy", "eval_seconds")
            },
        },
    )


if __name__ == "__main__":
    main()
