#!/usr/bin/env python3
"""Run the supplement Chapter 2 zero-shot GSM8K baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from cs336_alignment.metrics import parse_gsm8k_response
from cs336_alignment.prompt_templates import PROMPT_FORMAT_CHOICES, format_generation_prompt
from cs336_alignment.zero_shot import (
    generate_texts,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = "/data/a5-alignment/models/Llama-3.1-8B"
DEFAULT_INPUT_PATH = "data/gsm8k/test.jsonl"
DEFAULT_OUTPUT_PATH = (
    "/root/autodl-tmp/a5-alignment/runs/supplement/zero_shot/"
    "gsm8k_baseline_generations.jsonl"
)
DEFAULT_SUMMARY_PATH = ".agents/logs/ch2/gsm8k_baseline/summary.json"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMAT_CHOICES, default="zero_shot")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_gsm8k_examples(input_path: str, max_examples: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(input_path).open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_examples is not None and len(examples) >= max_examples:
                break
            row = json.loads(line)
            if "question" not in row or "answer" not in row:
                raise ValueError(f"Expected question and answer in {input_path}:{line_idx + 1}")
            gold_answer_text = row["answer"].split("####")[-1].strip()
            gold_answer = parse_gsm8k_response(gold_answer_text)
            if gold_answer is None:
                raise ValueError(f"Could not parse gold GSM8K answer in {input_path}:{line_idx + 1}")
            examples.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "gold_answer": gold_answer,
                    "source_row": line_idx,
                }
            )
    if not examples:
        raise ValueError(f"No GSM8K examples found in {input_path}")
    return examples


def format_gsm8k_instruction(question: str) -> str:
    return f"{question}\n\nAnswer:"


def summarize(records: list[dict[str, Any]], timing: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": "gsm8k",
        "input_path": args.input_path,
        "model": args.model,
        "prompt_format": args.prompt_format,
        "num_examples": len(records),
        "accuracy": mean(record["correct"] for record in records),
        "parse_failures": sum(record["parsed_prediction"] is None for record in records),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "stop": ["# Query:"],
            **timing,
        },
    }


def main() -> None:
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))
    examples = load_gsm8k_examples(args.input_path, args.max_examples)
    prompts = [
        format_generation_prompt(format_gsm8k_instruction(example["question"]), args.prompt_format)
        for example in examples
    ]
    outputs, timing = generate_texts(
        prompts=prompts,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    records: list[dict[str, Any]] = []
    for example, prompt, output in zip(examples, prompts, outputs):
        parsed_prediction = parse_gsm8k_response(output)
        correct = parsed_prediction == example["gold_answer"]
        records.append(
            {
                **example,
                "prompt": prompt,
                "model_output": output,
                "parsed_prediction": parsed_prediction,
                "correct": correct,
            }
        )

    summary = summarize(records, timing, args)
    write_jsonl(args.output_path, records)
    write_json(args.summary_path, summary)
    logger.info("wrote %s", args.output_path)
    logger.info("wrote %s", args.summary_path)
    logger.info("accuracy: %.4f", summary["accuracy"])
    logger.info("parse_failures: %s", summary["parse_failures"])
    logger.info("examples_per_second: %s", summary["generation"]["examples_per_second"])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
