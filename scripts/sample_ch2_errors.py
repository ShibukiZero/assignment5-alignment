"""Sample qualitative examples for supplement Chapter 2 writeup analysis."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MMLU_PATH = (
    "runs/supplement/ch2/"
    "mmlu_baseline/generations.jsonl"
)
DEFAULT_GSM8K_PATH = (
    "runs/supplement/ch2/"
    "gsm8k_baseline/generations.jsonl"
)
DEFAULT_ALPACA_PATH = (
    "artifacts/experiments/supplement/ch2/alpaca_eval_baseline/annotations.json"
)
DEFAULT_SAFETY_PATH = (
    "artifacts/experiments/supplement/ch2/"
    "simple_safety_tests_baseline/annotations.jsonl"
)
DEFAULT_OUTPUT_DIR = "artifacts/experiments/supplement/ch2/error_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample deterministic qualitative examples for Chapter 2."
    )
    parser.add_argument("--mmlu-path", default=DEFAULT_MMLU_PATH)
    parser.add_argument("--gsm8k-path", default=DEFAULT_GSM8K_PATH)
    parser.add_argument("--alpaca-path", default=DEFAULT_ALPACA_PATH)
    parser.add_argument("--safety-path", default=DEFAULT_SAFETY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def compact_mmlu(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject": record.get("subject"),
        "question": record.get("question"),
        "choices": {
            key: record.get(key)
            for key in ("A", "B", "C", "D")
            if key in record
        },
        "answer": record.get("answer"),
        "parsed_prediction": record.get("parsed_prediction"),
        "model_output": record.get("model_output"),
    }


def compact_gsm8k(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "parsed_prediction": record.get("parsed_prediction"),
        "model_output": record.get("model_output"),
    }


def compact_alpaca(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": record.get("instruction"),
        "reference_generator": record.get("generator_1"),
        "reference_output": record.get("output_1"),
        "baseline_generator": record.get("generator_2"),
        "baseline_output": record.get("output_2"),
        "preference": record.get("preference"),
        "raw_completion": record.get("raw_completion"),
    }


def compact_safety(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "harm_area": record.get("harm_area"),
        "category": record.get("category"),
        "prompt": record.get("prompts_final"),
        "output": record.get("output"),
        "judge_response": record.get("model_response"),
        "metrics": record.get("metrics"),
    }


def sample_records(
    records: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool = list(records)
    rng = random.Random(seed)
    if len(pool) <= count:
        return pool
    return rng.sample(pool, count)


def render_markdown(sections: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# Supplement Chapter 2 Error Samples",
        "",
        "These deterministic samples are for qualitative writeup analysis.",
        "",
    ]
    for title, records in sections.items():
        lines.extend([f"## {title}", ""])
        for index, record in enumerate(records, start=1):
            lines.append(f"### {index}")
            for key, value in record.items():
                rendered = json.dumps(value, ensure_ascii=False)
                lines.append(f"- **{key}:** {rendered}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    mmlu = read_jsonl(args.mmlu_path)
    gsm8k = read_jsonl(args.gsm8k_path)
    alpaca = read_json(args.alpaca_path)
    safety = read_jsonl(args.safety_path)

    sections = {
        "MMLU Incorrect": [
            compact_mmlu(record)
            for record in sample_records(
                (record for record in mmlu if not record.get("correct")),
                count=args.num_samples,
                seed=args.seed,
            )
        ],
        "GSM8K Incorrect": [
            compact_gsm8k(record)
            for record in sample_records(
                (record for record in gsm8k if not record.get("correct")),
                count=args.num_samples,
                seed=args.seed,
            )
        ],
        "AlpacaEval Dispreferred": [
            compact_alpaca(record)
            for record in sample_records(
                (record for record in alpaca if record.get("preference") == 1.0),
                count=args.num_samples,
                seed=args.seed,
            )
        ],
        "SimpleSafetyTests Unsafe": [
            compact_safety(record)
            for record in sample_records(
                (
                    record
                    for record in safety
                    if record.get("metrics", {}).get("safe") == 0.0
                ),
                count=args.num_samples,
                seed=args.seed,
            )
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for title, records in sections.items():
        filename = title.lower().replace(" ", "_") + ".json"
        write_json(output_dir / filename, records)
    (output_dir / "summary.md").write_text(render_markdown(sections), encoding="utf-8")

    counts = {title: len(records) for title, records in sections.items()}
    write_json(
        output_dir / "sample_config.json",
        {
            "seed": args.seed,
            "num_samples": args.num_samples,
            "paths": {
                "mmlu": args.mmlu_path,
                "gsm8k": args.gsm8k_path,
                "alpaca": args.alpaca_path,
                "safety": args.safety_path,
            },
            "counts": counts,
        },
    )


if __name__ == "__main__":
    main()
