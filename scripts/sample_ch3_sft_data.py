"""Sample instruction-tuning examples for supplement Chapter 3 analysis."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "/root/autodl-tmp/a5-alignment/datasets/ch3/train.jsonl.gz"
DEFAULT_OUTPUT_DIR = ".agents/logs/ch3/look_at_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample SFT training examples for Chapter 3."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def open_text(path: str | Path):
    input_path = Path(path)
    if input_path.suffix == ".gz":
        return gzip.open(input_path, "rt", encoding="utf-8")
    return input_path.open(encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open_text(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record or "response" not in record:
                raise ValueError(
                    f"Expected prompt and response in {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sample_records(
    records: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if num_samples >= len(records):
        return list(records)
    return rng.sample(records, num_samples)


def render_markdown(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Supplement Chapter 3 SFT Samples",
        "",
        "These examples were randomly sampled from the instruction-tuning train split.",
        "",
    ]
    for index, record in enumerate(samples, start=1):
        prompt = record["prompt"]
        response = record["response"]
        lines.extend(
            [
                f"## Sample {index}",
                "",
                "### Prompt",
                "",
                "```text",
                str(prompt),
                "```",
                "",
                "### Response",
                "",
                "```text",
                str(response),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    records = read_jsonl(args.input_path)
    samples = sample_records(
        records,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "samples.json", samples)
    write_json(
        output_dir / "sample_config.json",
        {
            "input_path": args.input_path,
            "num_records": len(records),
            "num_samples": len(samples),
            "seed": args.seed,
        },
    )
    (output_dir / "summary.md").write_text(
        render_markdown(samples),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
