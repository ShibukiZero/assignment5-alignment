"""Sample HH preference examples for supplement Chapter 5.2 analysis."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from cs336_alignment.dpo_data import load_hh_preference_data


DEFAULT_INPUT_DIR = "data/datasets/hh_rlhf"
DEFAULT_OUTPUT_DIR = "runs/logs/ch5/look_at_hh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Anthropic HH preference data, keep single-turn examples, "
            "and sample helpful/harmless pairs for writeup analysis."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="HH dataset root or a single HH JSONL/JSONL.GZ file.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-helpful", type=int, default=3)
    parser.add_argument("--num-harmless", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap after single-turn filtering, useful for quick smoke runs.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1800,
        help="Maximum characters shown per text field in summary.md.",
    )
    return parser.parse_args()


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def infer_preference_split(record: dict[str, Any]) -> str:
    split = str(record.get("split", "unknown"))
    if split in {"helpful", "harmless"}:
        return split

    source_path = str(record.get("source_path", "")).lower()
    source_file = str(record.get("source_file", "")).lower()
    source = f"{source_path}/{source_file}"
    if "harmless-base" in source:
        return "harmless"
    if (
        "helpful-base" in source
        or "helpful-online" in source
        or "helpful-rejection-sampled" in source
    ):
        return "helpful"
    return "unknown"


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "preference_split": infer_preference_split(record),
        "source_file": record.get("source_file"),
        "source_path": record.get("source_path"),
        "line_number": record.get("line_number"),
        "instruction": record.get("instruction"),
        "chosen": record.get("chosen"),
        "rejected": record.get("rejected"),
    }


def sample_records(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if count >= len(records):
        return list(records)
    return rng.sample(records, count)


def truncate_text(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def render_example(index: int, record: dict[str, Any], max_text_chars: int) -> list[str]:
    return [
        f"## Example {index}",
        "",
        f"- **preference_split:** {record['preference_split']}",
        f"- **source_path:** `{record['source_path']}`",
        f"- **line_number:** {record['line_number']}",
        "",
        "### Instruction",
        "",
        "```text",
        truncate_text(record["instruction"], max_text_chars),
        "```",
        "",
        "### Chosen",
        "",
        "```text",
        truncate_text(record["chosen"], max_text_chars),
        "```",
        "",
        "### Rejected",
        "",
        "```text",
        truncate_text(record["rejected"], max_text_chars),
        "```",
        "",
        "### Analysis Notes",
        "",
        "- Chosen vs rejected difference:",
        "- Do you agree with the annotator?",
        "",
    ]


def render_markdown(
    helpful_samples: list[dict[str, Any]],
    harmless_samples: list[dict[str, Any]],
    *,
    max_text_chars: int,
) -> str:
    lines = [
        "# Supplement Chapter 5.2 HH Preference Samples",
        "",
        "These examples were sampled from single-turn Anthropic HH preference pairs.",
        "Fill in the analysis notes after manually comparing each chosen/rejected pair.",
        "",
        "# Helpful Samples",
        "",
    ]

    for index, record in enumerate(helpful_samples, start=1):
        lines.extend(render_example(index, record, max_text_chars))

    lines.extend(["# Harmless Samples", ""])
    for index, record in enumerate(harmless_samples, start=1):
        lines.extend(render_example(index, record, max_text_chars))

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    records = [
        compact_record(record)
        for record in load_hh_preference_data(
            args.input_dir,
            max_records=args.max_records,
        )
    ]

    helpful = [
        record for record in records if record["preference_split"] == "helpful"
    ]
    harmless = [
        record for record in records if record["preference_split"] == "harmless"
    ]
    if len(helpful) < args.num_helpful:
        raise ValueError(
            f"Requested {args.num_helpful} helpful samples, but found {len(helpful)}."
        )
    if len(harmless) < args.num_harmless:
        raise ValueError(
            f"Requested {args.num_harmless} harmless samples, but found {len(harmless)}."
        )

    helpful_samples = sample_records(
        helpful,
        count=args.num_helpful,
        seed=args.seed,
    )
    harmless_samples = sample_records(
        harmless,
        count=args.num_harmless,
        seed=args.seed + 1,
    )
    samples = {
        "helpful": helpful_samples,
        "harmless": harmless_samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "samples.json", samples)
    write_json(
        output_dir / "sample_config.json",
        {
            "input_dir": args.input_dir,
            "output_dir": str(output_dir),
            "seed": args.seed,
            "num_helpful": args.num_helpful,
            "num_harmless": args.num_harmless,
            "max_records": args.max_records,
            "num_single_turn_records_loaded": len(records),
            "split_counts": dict(
                sorted(Counter(record["preference_split"] for record in records).items())
            ),
        },
    )
    (output_dir / "summary.md").write_text(
        render_markdown(
            helpful_samples,
            harmless_samples,
            max_text_chars=args.max_text_chars,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
