#!/usr/bin/env python3
"""Evaluate the zero-shot MATH baseline with vLLM.

This script implements the generation-and-grading loop for the
`math_baseline` problem. It reads validation examples, formats each question
with the R1-Zero prompt, generates model responses with vLLM, grades them with
the provided reward function, and writes per-example results plus a summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_MODEL_PATH = "models/Qwen2.5-Math-1.5B"
DEFAULT_INPUT_PATH = "data/MATH_like/competition_math_numeric/validation.jsonl"
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_LOG_DIR = "runs/logs/ch3/3_2_math_baseline"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot MATH-style baseline evaluation.")
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Validation JSONL path. Each row should contain `question` and `ground_truth`.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="JSONL path for per-example model generations and reward scores.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help=(
            "Optional JSON path for aggregate metrics. If omitted, the summary is written "
            "under --log-dir so it can be inspected before being archived to artifacts/."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=(
            "Directory for small experiment logs and summaries. Large per-example outputs "
            "should still go to the data disk via --output-path."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the Qwen2.5-Math-1.5B model directory, or a Hugging Face model id.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="R1-Zero prompt template path. It must contain `{question}`.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for quick remote smoke tests.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Set this above 1 only when intentionally sharding the model over multiple GPUs.",
    )
    return parser.parse_args()


def load_prompt_template(path: str) -> str:
    template = Path(path).read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"Prompt template must contain '{{question}}': {path}")
    return template


def load_examples(path: str, max_examples: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if max_examples is not None and len(examples) >= max_examples:
                break
            row = json.loads(line)
            question = row.get("question") or row.get("problem")
            ground_truth = row.get("ground_truth", row.get("answer"))
            if question is None or ground_truth is None:
                raise ValueError(
                    "Each example must contain `question`/`problem` and `ground_truth`/`answer`."
                )
            examples.append({**row, "question": question, "ground_truth": ground_truth})
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_examples = len(records)
    correct_format_correct_answer = sum(
        r["format_reward"] == 1.0 and r["answer_reward"] == 1.0 for r in records
    )
    correct_format_wrong_answer = sum(
        r["format_reward"] == 1.0 and r["answer_reward"] == 0.0 for r in records
    )
    wrong_format_wrong_answer = sum(
        r["format_reward"] == 0.0 and r["answer_reward"] == 0.0 for r in records
    )

    return {
        "num_examples": num_examples,
        "accuracy": mean(r["reward"] for r in records),
        "format_accuracy": mean(r["format_reward"] for r in records),
        "answer_accuracy": mean(r["answer_reward"] for r in records),
        "correct_format_correct_answer": correct_format_correct_answer,
        "correct_format_wrong_answer": correct_format_wrong_answer,
        "wrong_format_wrong_answer": wrong_format_wrong_answer,
    }


def default_summary_path(output_path: str, log_dir: str) -> Path:
    path = Path(output_path)
    return Path(log_dir) / f"{path.stem}.summary.json"


def main() -> None:
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))

    prompt_template = load_prompt_template(args.prompt_template)
    examples = load_examples(args.input_path, args.max_examples)
    prompts = [prompt_template.format(question=example["question"]) for example in examples]
    logger.info("loaded %d examples from %s", len(examples), args.input_path)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    outputs = llm.generate(prompts, sampling_params)

    records: list[dict[str, Any]] = []
    for example, prompt, output in tqdm(
        zip(examples, prompts, outputs), total=len(examples), desc="Scoring generations"
    ):
        response = output.outputs[0].text
        scores = r1_zero_reward_fn(response, example["ground_truth"])
        records.append(
            {
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "prompt": prompt,
                "response": response,
                "reward": scores["reward"],
                "format_reward": scores["format_reward"],
                "answer_reward": scores["answer_reward"],
                "source_example": example,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("wrote per-example results to %s", output_path)

    summary = summarize(records)
    summary.update(
        {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "model": args.model,
            "prompt_template": args.prompt_template,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "stop": "</answer>",
            "include_stop_str_in_output": True,
        }
    )
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else default_summary_path(args.output_path, args.log_dir)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote summary to %s", summary_path)

    for key, value in summary.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
