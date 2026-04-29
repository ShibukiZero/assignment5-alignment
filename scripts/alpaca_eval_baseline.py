#!/usr/bin/env python3
"""Generate supplement Chapter 2 zero-shot AlpacaEval outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from cs336_alignment.zero_shot import (
    format_supplement_prompt,
    generate_texts,
    write_json,
)


DEFAULT_MODEL = "/data/a5-alignment/models/Llama-3.1-8B"
DEFAULT_INPUT_PATH = "data/alpaca_eval/alpaca_eval.jsonl"
DEFAULT_OUTPUT_PATH = (
    "/root/autodl-tmp/a5-alignment/runs/supplement/zero_shot/"
    "alpaca_eval_baseline_outputs.json"
)
DEFAULT_SUMMARY_PATH = ".agents/logs/ch2/alpaca_eval_baseline/summary.json"
DEFAULT_GENERATOR = "llama-3.1-8b-base"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_alpaca_eval_examples(input_path: str, max_examples: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with Path(input_path).open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_examples is not None and len(examples) >= max_examples:
                break
            row = json.loads(line)
            if "instruction" not in row or "dataset" not in row:
                raise ValueError(f"Expected instruction and dataset in {input_path}:{line_idx + 1}")
            examples.append(row)
    if not examples:
        raise ValueError(f"No AlpacaEval examples found in {input_path}")
    return examples


def main() -> None:
    args = parse_args()
    logger.info("running %s", " ".join(sys.argv))
    examples = load_alpaca_eval_examples(args.input_path, args.max_examples)
    prompts = [format_supplement_prompt(example["instruction"]) for example in examples]
    outputs, timing = generate_texts(
        prompts=prompts,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    records: list[dict[str, Any]] = []
    for example, output in zip(examples, outputs):
        records.append(
            {
                "instruction": example["instruction"],
                "output": output,
                "generator": args.generator,
                "dataset": example["dataset"],
            }
        )

    summary = {
        "dataset": "alpaca_eval",
        "input_path": args.input_path,
        "output_path": args.output_path,
        "model": args.model,
        "generator": args.generator,
        "num_examples": len(records),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "stop": ["# Query:"],
            **timing,
        },
    }
    write_json(args.output_path, records)
    write_json(args.summary_path, summary)
    logger.info("wrote %s", args.output_path)
    logger.info("wrote %s", args.summary_path)
    logger.info("examples_per_second: %s", summary["generation"]["examples_per_second"])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
