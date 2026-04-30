"""Sample paired benchmark outputs for qualitative writeup analysis.

The script compares a baseline run directory against a candidate run directory.
It is intentionally benchmark-agnostic so it can be reused for SFT, DPO, GRPO,
or other checkpoints as long as the run directories contain the standard files
produced by the evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_BASELINE_DIR = ".agents/logs/data_disk_snapshots/zero_shot_official_prompt"
DEFAULT_CANDIDATE_DIR = ".agents/logs/data_disk_snapshots/ch3_sft_eval"
DEFAULT_OUTPUT_DIR = ".agents/logs/benchmark_samples/sft_vs_zero_shot"


JsonRecord = dict[str, Any]
KeyFn = Callable[[JsonRecord], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample deterministic paired benchmark examples from two evaluation "
            "run directories."
        )
    )
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-label", default="zero_shot")
    parser.add_argument("--candidate-label", default="sft")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Maximum characters to keep for each long text field.",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    records: list[JsonRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
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


def truncate_text(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if max_chars < 4 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def compact_text_fields(record: JsonRecord, max_chars: int) -> JsonRecord:
    return {key: truncate_text(value, max_chars) for key, value in record.items()}


def build_pair_index(records: Iterable[JsonRecord], key_fn: KeyFn) -> dict[str, JsonRecord]:
    index: dict[str, JsonRecord] = {}
    for record in records:
        key = key_fn(record)
        if key in index:
            raise ValueError(f"Duplicate pair key: {key}")
        index[key] = record
    return index


def sample_records(records: list[JsonRecord], *, count: int, seed: int) -> list[JsonRecord]:
    rng = random.Random(seed)
    if len(records) <= count:
        return list(records)
    return rng.sample(records, count)


def classify_binary_pair(
    baseline_correct: bool,
    candidate_correct: bool,
    baseline_label: str,
    candidate_label: str,
) -> str:
    if not baseline_correct and candidate_correct:
        return f"{baseline_label}_wrong_{candidate_label}_correct"
    if baseline_correct and not candidate_correct:
        return f"{baseline_label}_correct_{candidate_label}_wrong"
    if baseline_correct and candidate_correct:
        return "both_correct"
    return "both_wrong"


def alpaca_model_preference(record: JsonRecord) -> str:
    preference = record.get("preference")
    if preference == 2.0:
        return "model_preferred"
    if preference == 1.0:
        return "reference_preferred"
    if preference == 1.5:
        return "draw"
    return "unknown"


def safety_is_safe(record: JsonRecord) -> bool:
    return record.get("metrics", {}).get("safe") == 1.0


def key_from_source_row(record: JsonRecord) -> str:
    return str(record["source_row"])


def key_from_mmlu(record: JsonRecord) -> str:
    return f"{record.get('source_file')}:{record.get('source_row')}"


def key_from_instruction(record: JsonRecord) -> str:
    return str(record["instruction"])


def key_from_safety(record: JsonRecord) -> str:
    return str(record.get("id") or record.get("source_row"))


def paired_keys(
    baseline: Iterable[JsonRecord],
    candidate: Iterable[JsonRecord],
    key_fn: KeyFn,
) -> tuple[dict[str, JsonRecord], dict[str, JsonRecord], list[str]]:
    baseline_index = build_pair_index(baseline, key_fn)
    candidate_index = build_pair_index(candidate, key_fn)
    keys = sorted(set(baseline_index) & set(candidate_index))
    return baseline_index, candidate_index, keys


def sample_mmlu_or_gsm8k(
    baseline_records: list[JsonRecord],
    candidate_records: list[JsonRecord],
    *,
    key_fn: KeyFn,
    baseline_label: str,
    candidate_label: str,
    num_samples: int,
    seed: int,
    max_text_chars: int,
) -> tuple[dict[str, list[JsonRecord]], dict[str, int]]:
    baseline_index, candidate_index, keys = paired_keys(
        baseline_records, candidate_records, key_fn
    )
    buckets: dict[str, list[JsonRecord]] = defaultdict(list)

    for key in keys:
        baseline = baseline_index[key]
        candidate = candidate_index[key]
        bucket = classify_binary_pair(
            bool(baseline.get("correct")),
            bool(candidate.get("correct")),
            baseline_label,
            candidate_label,
        )
        sample = {
            "key": key,
            "question": baseline.get("question"),
            "subject": baseline.get("subject"),
            "options": baseline.get("options"),
            "gold_answer": baseline.get("answer") or baseline.get("gold_answer"),
            f"{baseline_label}_parsed_prediction": baseline.get("parsed_prediction"),
            f"{candidate_label}_parsed_prediction": candidate.get("parsed_prediction"),
            f"{baseline_label}_output": baseline.get("model_output"),
            f"{candidate_label}_output": candidate.get("model_output"),
        }
        buckets[bucket].append(compact_text_fields(sample, max_text_chars))

    population_counts = {bucket: len(records) for bucket, records in sorted(buckets.items())}
    samples = {
        bucket: sample_records(records, count=num_samples, seed=seed)
        for bucket, records in sorted(buckets.items())
    }
    return samples, population_counts


def sample_alpaca_eval(
    baseline_records: list[JsonRecord],
    candidate_records: list[JsonRecord],
    *,
    baseline_label: str,
    candidate_label: str,
    num_samples: int,
    seed: int,
    max_text_chars: int,
) -> tuple[dict[str, list[JsonRecord]], dict[str, int]]:
    baseline_index, candidate_index, keys = paired_keys(
        baseline_records, candidate_records, key_from_instruction
    )
    buckets: dict[str, list[JsonRecord]] = defaultdict(list)

    for key in keys:
        baseline = baseline_index[key]
        candidate = candidate_index[key]
        baseline_pref = alpaca_model_preference(baseline)
        candidate_pref = alpaca_model_preference(candidate)
        bucket = f"{baseline_label}_{baseline_pref}__{candidate_label}_{candidate_pref}"
        sample = {
            "instruction": key,
            "reference_generator": candidate.get("generator_1"),
            "reference_output": candidate.get("output_1"),
            f"{baseline_label}_generator": baseline.get("generator_2"),
            f"{baseline_label}_preference": baseline.get("preference"),
            f"{baseline_label}_preference_label": baseline_pref,
            f"{baseline_label}_output": baseline.get("output_2"),
            f"{candidate_label}_generator": candidate.get("generator_2"),
            f"{candidate_label}_preference": candidate.get("preference"),
            f"{candidate_label}_preference_label": candidate_pref,
            f"{candidate_label}_output": candidate.get("output_2"),
        }
        buckets[bucket].append(compact_text_fields(sample, max_text_chars))

    population_counts = {bucket: len(records) for bucket, records in sorted(buckets.items())}
    samples = {
        bucket: sample_records(records, count=num_samples, seed=seed)
        for bucket, records in sorted(buckets.items())
    }
    return samples, population_counts


def sample_safety(
    baseline_records: list[JsonRecord],
    candidate_records: list[JsonRecord],
    *,
    baseline_label: str,
    candidate_label: str,
    num_samples: int,
    seed: int,
    max_text_chars: int,
) -> tuple[dict[str, list[JsonRecord]], dict[str, int]]:
    baseline_index, candidate_index, keys = paired_keys(
        baseline_records, candidate_records, key_from_safety
    )
    buckets: dict[str, list[JsonRecord]] = defaultdict(list)

    for key in keys:
        baseline = baseline_index[key]
        candidate = candidate_index[key]
        baseline_safe = safety_is_safe(baseline)
        candidate_safe = safety_is_safe(candidate)
        bucket = classify_binary_pair(
            baseline_safe,
            candidate_safe,
            baseline_label,
            candidate_label,
        ).replace("correct", "safe").replace("wrong", "unsafe")
        sample = {
            "key": key,
            "harm_area": baseline.get("harm_area"),
            "category": baseline.get("category"),
            "prompt": baseline.get("prompts_final"),
            f"{baseline_label}_safe": baseline_safe,
            f"{baseline_label}_judge_response": baseline.get("model_response"),
            f"{baseline_label}_output": baseline.get("output"),
            f"{candidate_label}_safe": candidate_safe,
            f"{candidate_label}_judge_response": candidate.get("model_response"),
            f"{candidate_label}_output": candidate.get("output"),
        }
        buckets[bucket].append(compact_text_fields(sample, max_text_chars))

    population_counts = {bucket: len(records) for bucket, records in sorted(buckets.items())}
    samples = {
        bucket: sample_records(records, count=num_samples, seed=seed)
        for bucket, records in sorted(buckets.items())
    }
    return samples, population_counts


def render_markdown(samples: dict[str, dict[str, list[JsonRecord]]]) -> str:
    lines = [
        "# Paired Benchmark Samples",
        "",
        "These deterministic samples compare a baseline run against a candidate run.",
        "",
    ]
    for benchmark, buckets in samples.items():
        lines.extend([f"## {benchmark}", ""])
        for bucket, records in buckets.items():
            lines.extend([f"### {bucket}", ""])
            if not records:
                lines.extend(["No examples in this bucket.", ""])
                continue
            for index, record in enumerate(records, start=1):
                lines.append(f"#### Sample {index}")
                for key, value in record.items():
                    rendered = json.dumps(value, ensure_ascii=False)
                    lines.append(f"- **{key}:** {rendered}")
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)

    mmlu_samples, mmlu_population_counts = sample_mmlu_or_gsm8k(
        read_jsonl(baseline_dir / "mmlu_generations.jsonl"),
        read_jsonl(candidate_dir / "mmlu_generations.jsonl"),
        key_fn=key_from_mmlu,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        num_samples=args.num_samples,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
    )
    gsm8k_samples, gsm8k_population_counts = sample_mmlu_or_gsm8k(
        read_jsonl(baseline_dir / "gsm8k_generations.jsonl"),
        read_jsonl(candidate_dir / "gsm8k_generations.jsonl"),
        key_fn=key_from_source_row,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        num_samples=args.num_samples,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
    )
    alpaca_samples, alpaca_population_counts = sample_alpaca_eval(
        read_json(baseline_dir / "annotations.json"),
        read_json(candidate_dir / "annotations.json"),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        num_samples=args.num_samples,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
    )
    safety_samples, safety_population_counts = sample_safety(
        read_jsonl(baseline_dir / "simple_safety_tests_annotations.jsonl"),
        read_jsonl(candidate_dir / "simple_safety_tests_annotations.jsonl"),
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        num_samples=args.num_samples,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
    )
    samples = {
        "mmlu": mmlu_samples,
        "gsm8k": gsm8k_samples,
        "alpaca_eval": alpaca_samples,
        "simple_safety_tests": safety_samples,
    }

    sample_counts = {
        benchmark: {bucket: len(records) for bucket, records in buckets.items()}
        for benchmark, buckets in samples.items()
    }
    population_counts = {
        "mmlu": mmlu_population_counts,
        "gsm8k": gsm8k_population_counts,
        "alpaca_eval": alpaca_population_counts,
        "simple_safety_tests": safety_population_counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "samples.json", samples)
    write_json(
        output_dir / "sample_config.json",
        {
            "baseline_dir": str(baseline_dir),
            "candidate_dir": str(candidate_dir),
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "max_text_chars": args.max_text_chars,
            "sample_counts": sample_counts,
            "population_counts": population_counts,
        },
    )
    (output_dir / "summary.md").write_text(
        render_markdown(samples),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
