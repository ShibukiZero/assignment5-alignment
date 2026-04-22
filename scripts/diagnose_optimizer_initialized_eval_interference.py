#!/usr/bin/env python3
"""Compare first vLLM eval with and without an attached optimizer in unified runtime."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from torch.optim import AdamW
from vllm import SamplingParams

from cs336_alignment.backend_lifecycle import BackendLifecycleManager
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_MODEL = "/root/autodl-tmp/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_VAL_PATH = (
    "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric_noisy/validation.jsonl"
)
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = ".agents/logs/diagnose_optimizer_initialized_eval_interference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether attaching an optimizer changes the first unified-runtime eval."
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
    parser.add_argument("--sample-dump-count", type=int, default=5)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_path(
    *,
    manager: BackendLifecycleManager,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    prompts = [prompt_template.format(question=get_question(example)) for example in examples]
    state_before_eval = manager.debug_state()
    llm = manager.enter_inference_phase(sync_weights=True)
    outputs = llm.generate(prompts, sampling_params)
    state_after_eval = manager.debug_state()

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
    manager.enter_training_phase()

    summary = {
        "num_examples": len(records),
        "reward": mean(record["reward"] for record in records),
        "format_accuracy": mean(record["format_reward"] for record in records),
        "answer_accuracy": mean(record["answer_reward"] for record in records),
        "state_before_eval": state_before_eval,
        "state_after_eval": state_after_eval,
    }
    return {"summary": summary, "records": records}


def build_manager(args: argparse.Namespace) -> BackendLifecycleManager:
    return BackendLifecycleManager.from_defaults(
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


def run_no_optimizer_runtime(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    manager = build_manager(args)
    manager.initialize_rl_runtime()
    manager.enter_training_phase()
    return evaluate_path(
        manager=manager,
        examples=examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )


def run_with_optimizer_runtime(
    args: argparse.Namespace,
    *,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    manager = build_manager(args)
    manager.initialize_rl_runtime(
        optimizer_factory=lambda parameters: AdamW(parameters, lr=args.learning_rate)
    )
    manager.enter_training_phase()
    return evaluate_path(
        manager=manager,
        examples=examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )


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

    no_optimizer = run_no_optimizer_runtime(
        args,
        examples=eval_examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )
    with_optimizer = run_with_optimizer_runtime(
        args,
        examples=eval_examples,
        prompt_template=prompt_template,
        sampling_params=sampling_params,
    )

    write_json(log_dir / "no_optimizer_summary.json", no_optimizer["summary"])
    write_json(log_dir / "with_optimizer_summary.json", with_optimizer["summary"])
    write_jsonl(log_dir / "no_optimizer_records.jsonl", no_optimizer["records"])
    write_jsonl(log_dir / "with_optimizer_records.jsonl", with_optimizer["records"])
    write_jsonl(
        log_dir / "no_optimizer_sample_records.jsonl",
        no_optimizer["records"][: args.sample_dump_count],
    )
    write_jsonl(
        log_dir / "with_optimizer_sample_records.jsonl",
        with_optimizer["records"][: args.sample_dump_count],
    )
    write_json(
        log_dir / "comparison_summary.json",
        {
            "no_optimizer_runtime": no_optimizer["summary"],
            "with_optimizer_runtime": with_optimizer["summary"],
            "delta": {
                key: with_optimizer["summary"][key] - no_optimizer["summary"][key]
                for key in ("reward", "format_accuracy", "answer_accuracy")
            },
        },
    )


if __name__ == "__main__":
    main()
