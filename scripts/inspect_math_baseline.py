#!/usr/bin/env python3
"""Sample math baseline generations for writeup inspection.

The full generation JSONL should stay on the data disk. This script reads that
large file, selects a small number of examples from each reward category, and
writes compact inspection artifacts under `.agents/logs/` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "/root/autodl-tmp/a5-alignment/runs/math_baseline_competition_math_numeric.jsonl"
DEFAULT_LOG_DIR = ".agents/logs/ch3/3_2_math_baseline"
BUCKETS = (
    "correct_format_correct_answer",
    "correct_format_wrong_answer",
    "wrong_format_wrong_answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect sampled math baseline generations.")
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Full per-example baseline JSONL path on the data disk.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for compact inspection outputs.",
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=10,
        help="Number of examples to keep from each reward category.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional output JSONL path. Defaults to <log-dir>/category_samples.jsonl.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional Markdown output path. Defaults to <log-dir>/category_samples.md.",
    )
    return parser.parse_args()


def bucket_name(record: dict[str, Any]) -> str | None:
    format_reward = record["format_reward"]
    answer_reward = record["answer_reward"]
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct_format_correct_answer"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "correct_format_wrong_answer"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "wrong_format_wrong_answer"
    return None


def compact_record(record: dict[str, Any], bucket: str, index: int) -> dict[str, Any]:
    source = record.get("source_example", {})
    return {
        "index": index,
        "bucket": bucket,
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "response": record["response"],
        "reward": record["reward"],
        "format_reward": record["format_reward"],
        "answer_reward": record["answer_reward"],
        "level": source.get("level"),
        "type": source.get("type"),
    }


def write_jsonl(path: Path, samples: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for bucket in BUCKETS:
            for record in samples[bucket]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(path: Path, samples: dict[str, list[dict[str, Any]]]) -> None:
    lines: list[str] = ["# Math Baseline Category Samples", ""]
    for bucket in BUCKETS:
        lines.extend([f"## {bucket}", ""])
        for record in samples[bucket]:
            response = record["response"].strip()
            lines.extend(
                [
                    f"### Sample {record['index']}",
                    "",
                    f"- Reward: {record['reward']}",
                    f"- Format reward: {record['format_reward']}",
                    f"- Answer reward: {record['answer_reward']}",
                    f"- Ground truth: `{record['ground_truth']}`",
                    f"- Level: `{record['level']}`",
                    f"- Type: `{record['type']}`",
                    "",
                    "**Question:**",
                    "",
                    record["question"],
                    "",
                    "**Response:**",
                    "",
                    "```text",
                    response,
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else log_dir / "category_samples.jsonl"
    output_md = Path(args.output_md) if args.output_md else log_dir / "category_samples.md"

    samples: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    seen = 0
    with Path(args.input_path).open(encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            record = json.loads(line)
            seen += 1
            bucket = bucket_name(record)
            if bucket is None:
                continue
            if len(samples[bucket]) < args.samples_per_bucket:
                samples[bucket].append(compact_record(record, bucket, index))
            if all(len(samples[bucket]) >= args.samples_per_bucket for bucket in BUCKETS):
                break

    write_jsonl(output_jsonl, samples)
    write_markdown(output_md, samples)

    summary = {
        "input_path": args.input_path,
        "seen_records": seen,
        "samples_per_bucket": args.samples_per_bucket,
        "output_jsonl": str(output_jsonl),
        "output_md": str(output_md),
        "bucket_counts": {bucket: len(samples[bucket]) for bucket in BUCKETS},
    }
    summary_path = log_dir / "category_samples.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
