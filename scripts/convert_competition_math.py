#!/usr/bin/env python3
"""Convert `jeggers/competition_math` into Assignment-5-style JSONL files.

The script writes:
- train.jsonl / validation.jsonl for RL and evaluation.
- sft.jsonl for supervised fine-tuning on reasoning traces.

It is intended for self-hosted environments where the course MATH files under
`/data/a5-alignment/MATH` are unavailable.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_OUTPUT_DIR = "data/MATH_like/competition_math_numeric"
DEFAULT_CACHE_DIR = "data/hf-cache"
DEFAULT_PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
DATA_FILE_SUFFIXES = {".parquet", ".jsonl", ".json"}
ANSWER_BLOCK_RE = re.compile(r"(?s)(<answer>)(.*?)(</answer>)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert competition_math into MATH-like train/validation/SFT JSONL files."
    )
    parser.add_argument(
        "--source",
        default="jeggers/competition_math",
        help=(
            "Hugging Face dataset id or local dataset directory. "
            "For a local huggingface-cli download, try the downloaded directory path."
        ),
    )
    parser.add_argument(
        "--config",
        default="numeric",
        help="Dataset config to load. Start with `numeric`; try `original` after the pipeline is stable.",
    )
    parser.add_argument("--train-split", default="train", help="Input split to write as train.jsonl.")
    parser.add_argument(
        "--validation-split",
        default="test",
        help="Input split to write as validation.jsonl. The public dataset commonly uses `test`.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_PATH,
        help="R1-Zero prompt template path. It must contain `{question}`.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap for train examples, useful for a quick remote smoke test.",
    )
    parser.add_argument(
        "--max-validation",
        type=int,
        default=None,
        help="Optional cap for validation examples, useful for a quick remote smoke test.",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Only write train.jsonl and validation.jsonl; skip sft.jsonl.",
    )
    parser.add_argument(
        "--prefer-extracted-solution",
        action="store_true",
        help=(
            "Use extracted_solution before boxed answers. This is easier for numeric-only runs, "
            "but may lose exact LaTeX forms such as fractions."
        ),
    )
    parser.add_argument(
        "--embed-ground-truth-in-sft",
        action="store_true",
        help="Include `ground_truth` and `question` fields in each SFT row for later filtering.",
    )
    parser.add_argument(
        "--sft-corruption-rate",
        type=float,
        default=0.0,
        help=(
            "Fraction of SFT rows whose final <answer> block should be replaced with an incorrect "
            "answer while preserving response format."
        ),
    )
    parser.add_argument(
        "--sft-corruption-seed",
        type=int,
        default=0,
        help="Random seed used when corrupting SFT answers.",
    )
    return parser.parse_args()


def find_local_data_files(source_path: Path, config: str) -> dict[str, list[str]] | None:
    candidate_files = [
        path
        for path in source_path.rglob("*")
        if path.is_file() and path.suffix.lower() in DATA_FILE_SUFFIXES
    ]
    if not candidate_files:
        return None

    config_filtered = [
        path
        for path in candidate_files
        if config in path.parts or config in path.name
    ]
    files = config_filtered or candidate_files

    data_files: dict[str, list[str]] = {}
    for split in ("train", "validation", "val", "test"):
        split_files = [
            str(path)
            for path in files
            if split in path.parts or path.name.startswith(f"{split}-") or path.stem == split
        ]
        if split_files:
            normalized_split = "validation" if split == "val" else split
            data_files.setdefault(normalized_split, []).extend(sorted(split_files))
    return data_files or None


def dataset_loader_name(data_files: dict[str, list[str]]) -> str:
    suffixes = {Path(path).suffix.lower() for paths in data_files.values() for path in paths}
    if suffixes == {".parquet"}:
        return "parquet"
    if suffixes <= {".json", ".jsonl"}:
        return "json"
    raise SystemExit(f"Mixed local dataset file types are not supported: {sorted(suffixes)}")


def load_hf_dataset(source: str, config: str, cache_dir: str):
    try:
        from datasets import Dataset, DatasetDict, load_dataset
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "This script needs `datasets` and `pandas`. "
            "Install them with `pip install -U datasets pandas`."
        ) from exc

    source_path = Path(source)
    if source_path.exists():
        local_data_files = find_local_data_files(source_path, config)
        if local_data_files:
            loader_name = dataset_loader_name(local_data_files)
            print(f"Loading local {loader_name} files from {source_path}")

            ds_dict = {}
            for split_name, paths in local_data_files.items():
                frames = []
                for p in paths:
                    p = str(p)
                    if p.endswith(".parquet"):
                        frames.append(pd.read_parquet(p))
                    elif p.endswith(".jsonl"):
                        frames.append(pd.read_json(p, lines=True))
                    elif p.endswith(".json"):
                        frames.append(pd.read_json(p))
                    else:
                        raise SystemExit(f"Unsupported local file: {p}")

                df = pd.concat(frames, ignore_index=True)
                ds_dict[split_name] = Dataset.from_pandas(df, preserve_index=False)

            return DatasetDict(ds_dict)

        print(f"No parquet/json/jsonl files found under {source_path}; trying dataset builder path.")
        return load_dataset(str(source_path), config, cache_dir=cache_dir)

    return load_dataset(source, config, cache_dir=cache_dir)


def iter_limited(split, limit: int | None) -> Iterable[dict[str, Any]]:
    count = 0
    for row in split:
        if limit is not None and count >= limit:
            break
        count += 1
        yield dict(row)


def extract_boxed_answers(text: str) -> list[str]:
    """Extract top-level contents from LaTeX \\boxed{...} expressions."""
    answers: list[str] = []
    marker = r"\boxed"
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index == -1:
            break
        brace_index = text.find("{", marker_index + len(marker))
        if brace_index == -1:
            start = marker_index + len(marker)
            continue

        depth = 0
        for index in range(brace_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[brace_index + 1 : index].strip())
                    start = index + 1
                    break
        else:
            break
    return [answer for answer in answers if answer]


def stringify_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    if isinstance(value, int):
        return str(value)

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None

    # Common artifact from numeric datasets: keep integers exact when possible.
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]
    return text


def choose_ground_truth(row: dict[str, Any], prefer_extracted_solution: bool) -> str | None:
    solution = str(row.get("solution") or "")
    boxed_answers = extract_boxed_answers(solution)
    boxed_answer = boxed_answers[-1] if boxed_answers else None
    extracted_answer = stringify_answer(row.get("extracted_solution"))

    if prefer_extracted_solution:
        return extracted_answer or boxed_answer
    return boxed_answer or extracted_answer


def make_eval_record(row: dict[str, Any], split_name: str, prefer_extracted_solution: bool) -> dict[str, Any] | None:
    problem = str(row.get("problem") or "").strip()
    ground_truth = choose_ground_truth(row, prefer_extracted_solution=prefer_extracted_solution)
    if not problem or ground_truth is None:
        return None

    record: dict[str, Any] = {
        "question": problem,
        "ground_truth": ground_truth,
        "problem": problem,
        "answer": ground_truth,
        "solution": str(row.get("solution") or "").strip(),
        "level": row.get("level"),
        "type": row.get("type"),
        "source": "jeggers/competition_math",
        "source_split": split_name,
    }
    return record


def clean_reasoning_trace(solution: str, final_answer: str) -> str:
    reasoning = solution.strip()
    if not reasoning:
        return f"We need solve the problem and provide the final answer."

    # Keep the source solution mostly intact. The final answer is also placed in
    # `<answer>` so the downstream reward parser sees the expected format.
    return reasoning.replace("\r\n", "\n").strip()


def replace_answer_block(response: str, new_answer: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)} {new_answer} {match.group(3)}"

    updated_response, num_subs = ANSWER_BLOCK_RE.subn(repl, response, count=1)
    if num_subs != 1:
        raise ValueError("Could not find exactly one <answer>...</answer> block to replace.")
    return updated_response


def choose_incorrect_answer(
    candidates: list[str],
    response: str,
    ground_truth: str,
    rng: random.Random,
) -> str:
    shuffled_candidates = list(candidates)
    rng.shuffle(shuffled_candidates)
    for candidate in shuffled_candidates:
        if candidate == ground_truth:
            continue
        corrupted_response = replace_answer_block(response, candidate)
        scores = r1_zero_reward_fn(corrupted_response, ground_truth)
        if scores["answer_reward"] == 0.0:
            return candidate
    raise ValueError("Failed to sample an incorrect replacement answer.")


def make_sft_record(
    eval_record: dict[str, Any],
    prompt_template: str,
    *,
    embed_ground_truth: bool,
) -> dict[str, Any]:
    question = eval_record["question"]
    final_answer = eval_record["ground_truth"]
    prompt = prompt_template.format(question=question)
    reasoning = clean_reasoning_trace(eval_record.get("solution", ""), final_answer)
    response = f"{reasoning}\n</think> <answer> {final_answer} </answer>"
    record: dict[str, str] = {"prompt": prompt, "response": response}
    if embed_ground_truth:
        record["question"] = question
        record["ground_truth"] = final_answer
        if eval_record.get("source_split") is not None:
            record["source_split"] = str(eval_record["source_split"])
    return record


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_split(
    split,
    split_name: str,
    output_path: Path,
    max_examples: int | None,
    prefer_extracted_solution: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    for row in iter_limited(split, max_examples):
        record = make_eval_record(
            row,
            split_name=split_name,
            prefer_extracted_solution=prefer_extracted_solution,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)

    written = write_jsonl(output_path, records)
    print(f"Wrote {written} records to {output_path} (skipped {skipped}).")
    return records


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.sft_corruption_rate <= 1.0:
        raise SystemExit("--sft-corruption-rate must lie in [0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")
    if "{question}" not in prompt_template:
        raise SystemExit(f"Prompt template must contain {{question}}: {args.prompt_template}")

    dataset = load_hf_dataset(args.source, args.config, args.cache_dir)
    missing_splits = [split for split in (args.train_split, args.validation_split) if split not in dataset]
    if missing_splits:
        available = ", ".join(dataset.keys())
        raise SystemExit(f"Missing split(s) {missing_splits}; available splits: {available}")

    train_records = convert_split(
        dataset[args.train_split],
        split_name=args.train_split,
        output_path=output_dir / "train.jsonl",
        max_examples=args.max_train,
        prefer_extracted_solution=args.prefer_extracted_solution,
    )
    validation_records = convert_split(
        dataset[args.validation_split],
        split_name=args.validation_split,
        output_path=output_dir / "validation.jsonl",
        max_examples=args.max_validation,
        prefer_extracted_solution=args.prefer_extracted_solution,
    )

    if not args.skip_sft:
        sft_records = [
            make_sft_record(
                record,
                prompt_template,
                embed_ground_truth=args.embed_ground_truth_in_sft,
            )
            for record in train_records
        ]
        if args.sft_corruption_rate > 0.0:
            rng = random.Random(args.sft_corruption_seed)
            candidate_answers = [str(record["ground_truth"]) for record in train_records]
            num_to_corrupt = int(round(len(sft_records) * args.sft_corruption_rate))
            corrupt_indices = set(rng.sample(range(len(sft_records)), num_to_corrupt))
            for index in corrupt_indices:
                ground_truth = str(train_records[index]["ground_truth"])
                replacement = choose_incorrect_answer(
                    candidates=candidate_answers,
                    response=sft_records[index]["response"],
                    ground_truth=ground_truth,
                    rng=rng,
                )
                sft_records[index]["response"] = replace_answer_block(
                    sft_records[index]["response"], replacement
                )
                if args.embed_ground_truth_in_sft:
                    sft_records[index]["was_corrupted"] = True
            if args.embed_ground_truth_in_sft:
                for index, row in enumerate(sft_records):
                    row.setdefault("was_corrupted", False)
        written = write_jsonl(output_dir / "sft.jsonl", sft_records)
        print(f"Wrote {written} records to {output_dir / 'sft.jsonl'}.")

    manifest = {
        "source": args.source,
        "config": args.config,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "wrote_sft": not args.skip_sft,
        "embed_ground_truth_in_sft": args.embed_ground_truth_in_sft,
        "sft_corruption_rate": args.sft_corruption_rate,
        "sft_corruption_seed": args.sft_corruption_seed,
        "answer_policy": "extracted_solution_first"
        if args.prefer_extracted_solution
        else "boxed_answer_first",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest to {output_dir / 'manifest.json'}.")


if __name__ == "__main__":
    main()
