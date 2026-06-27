#!/usr/bin/env python3
"""Run the supplement Chapter 2 zero-shot MMLU baseline."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from cs336_alignment.metrics import parse_mmlu_response
from cs336_alignment.prompt_templates import PROMPT_FORMAT_CHOICES, format_generation_prompt
from cs336_alignment.zero_shot import (
    generate_texts,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = "models/Llama-3.1-8B"
DEFAULT_MMLU_DIR = "data/mmlu"
DEFAULT_OUTPUT_PATH = (
    "runs/supplement/zero_shot/"
    "mmlu_baseline_generations.jsonl"
)
DEFAULT_SUMMARY_PATH = "runs/logs/ch2/mmlu_baseline/summary.json"

logger = logging.getLogger(__name__)


MMLU_INSTRUCTION_TEMPLATE = """Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}

A. {option_a}

B. {option_b}

C. {option_c}

D. {option_d}

Answer:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mmlu-dir", default=DEFAULT_MMLU_DIR)
    parser.add_argument("--split", default="test", choices=["dev", "val", "test"])
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMAT_CHOICES, default="zero_shot")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def subject_from_path(path: Path, split: str) -> str:
    suffix = f"_{split}"
    subject = path.stem
    if subject.endswith(suffix):
        subject = subject[: -len(suffix)]
    return subject.replace("_", " ")


def load_mmlu_examples(mmlu_dir: str, split: str, max_examples: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    split_dir = Path(mmlu_dir) / split
    for csv_path in sorted(split_dir.glob(f"*_{split}.csv")):
        subject = subject_from_path(csv_path, split)
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
                if len(row) != 6:
                    raise ValueError(f"Expected 6 columns in {csv_path}:{row_idx + 1}, got {len(row)}")
                question, option_a, option_b, option_c, option_d, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": [option_a, option_b, option_c, option_d],
                        "answer": answer.strip().upper(),
                        "source_file": str(csv_path),
                        "source_row": row_idx,
                    }
                )
    if not examples:
        raise ValueError(f"No MMLU examples found under {split_dir}")
    return examples


def format_mmlu_instruction(example: dict[str, Any]) -> str:
    return MMLU_INSTRUCTION_TEMPLATE.format(
        subject=example["subject"],
        question=example["question"],
        option_a=example["options"][0],
        option_b=example["options"][1],
        option_c=example["options"][2],
        option_d=example["options"][3],
    )


def summarize(records: list[dict[str, Any]], timing: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    subject_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        subject_records[record["subject"]].append(record)

    by_subject = {
        subject: {
            "num_examples": len(items),
            "accuracy": mean(item["correct"] for item in items),
            "parse_failures": sum(item["parsed_prediction"] is None for item in items),
        }
        for subject, items in sorted(subject_records.items())
    }
    return {
        "dataset": "mmlu",
        "split": args.split,
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
        "by_subject": by_subject,
    }


def main() -> None:
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))
    examples = load_mmlu_examples(args.mmlu_dir, args.split, args.max_examples)
    prompts = [
        format_generation_prompt(format_mmlu_instruction(example), args.prompt_format)
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
        parsed_prediction = parse_mmlu_response(example, output)
        correct = parsed_prediction == example["answer"]
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
