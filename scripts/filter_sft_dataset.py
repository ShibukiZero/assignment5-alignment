#!/usr/bin/env python3
"""Filter SFT examples by scoring their target responses against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_DATA_DIR = "/root/autodl-tmp/a5-alignment/MATH_like/competition_math_numeric"
DEFAULT_SFT_PATH = f"{DEFAULT_DATA_DIR}/sft.jsonl"
DEFAULT_TRAIN_PATH = f"{DEFAULT_DATA_DIR}/train.jsonl"
DEFAULT_OUTPUT_PATH = f"{DEFAULT_DATA_DIR}/filtered_sft.jsonl"
DEFAULT_SUMMARY_PATH = f"{DEFAULT_DATA_DIR}/filtered_sft_summary.json"
DEFAULT_BAD_EXAMPLES_PATH = f"{DEFAULT_DATA_DIR}/filtered_sft_bad_examples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair SFT examples with the source train split, score them with "
            "r1_zero_reward_fn, and keep only answer-correct examples."
        )
    )
    parser.add_argument("--sft-path", default=DEFAULT_SFT_PATH)
    parser.add_argument(
        "--train-path",
        default=None,
        help=(
            "Optional train.jsonl used to recover ground truth for legacy SFT files "
            "that only contain prompt/response."
        ),
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--bad-examples-path", default=DEFAULT_BAD_EXAMPLES_PATH)
    parser.add_argument(
        "--max-bad-examples",
        type=int,
        default=50,
        help="Maximum number of incorrect examples to dump for inspection.",
    )
    parser.add_argument(
        "--allow-prompt-mismatch",
        action="store_true",
        help=(
            "Skip the prompt/question consistency check. Leave this off unless you "
            "know the SFT file was reordered or post-processed."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prompt_matches_question(prompt: str, question: str) -> bool:
    return question in prompt


def main() -> None:
    args = parse_args()
    sft_rows = read_jsonl(args.sft_path)
    train_rows = read_jsonl(args.train_path) if args.train_path is not None else None

    needs_pairing = any("ground_truth" not in row for row in sft_rows)
    if needs_pairing and train_rows is None:
        raise ValueError(
            "SFT rows do not contain ground_truth, so --train-path is required for positional pairing."
        )
    if train_rows is not None and len(sft_rows) != len(train_rows):
        raise ValueError(
            "SFT and train files must have the same number of rows for positional pairing. "
            f"Got len(sft)={len(sft_rows)} and len(train)={len(train_rows)}."
        )

    filtered_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    num_format_correct = 0
    num_answer_correct = 0

    row_iterator = (
        zip(sft_rows, train_rows, strict=True)
        if train_rows is not None
        else ((sft_row, None) for sft_row in sft_rows)
    )
    for index, (sft_row, train_row) in enumerate(row_iterator):
        prompt = sft_row["prompt"]
        response = sft_row["response"]
        question = sft_row.get("question")
        ground_truth = sft_row.get("ground_truth")
        source_split = sft_row.get("source_split")

        if ground_truth is None and train_row is not None:
            question = train_row["question"]
            ground_truth = train_row["ground_truth"]
            source_split = train_row.get("source_split")

        if question is not None and not args.allow_prompt_mismatch and not prompt_matches_question(prompt, question):
            raise ValueError(
                "Prompt/question mismatch at row "
                f"{index}. This suggests the SFT file is not aligned with train.jsonl."
            )
        if ground_truth is None:
            raise ValueError(f"Missing ground_truth at row {index}.")

        scores = r1_zero_reward_fn(response, ground_truth)
        num_format_correct += int(scores["format_reward"] == 1.0)
        num_answer_correct += int(scores["answer_reward"] == 1.0)

        scored_row = {
            "prompt": prompt,
            "response": response,
            "question": question,
            "ground_truth": ground_truth,
            "source_split": source_split,
            "scores": scores,
        }

        if scores["answer_reward"] == 1.0:
            filtered_rows.append(scored_row)
        elif len(bad_rows) < args.max_bad_examples:
            bad_rows.append(scored_row)

    summary = {
        "sft_path": args.sft_path,
        "train_path": args.train_path,
        "output_path": args.output_path,
        "total_examples": len(sft_rows),
        "filtered_examples": len(filtered_rows),
        "removed_examples": len(sft_rows) - len(filtered_rows),
        "format_accuracy": num_format_correct / len(sft_rows) if sft_rows else 0.0,
        "answer_accuracy": num_answer_correct / len(sft_rows) if sft_rows else 0.0,
        "pairing_mode": "positional_zip" if train_rows is not None else "embedded_ground_truth",
        "prompt_check_enabled": not args.allow_prompt_mismatch,
    }

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    bad_examples_path = Path(args.bad_examples_path)

    write_jsonl(output_path, filtered_rows)
    write_json(summary_path, summary)
    write_jsonl(bad_examples_path, bad_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
