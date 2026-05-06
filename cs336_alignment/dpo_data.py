from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO


HH_TRAIN_COLLECTIONS = (
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
)

_TURN_RE = re.compile(r"\n\n(Human|Assistant):")


def _open_text(path: str | Path) -> TextIO:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        return gzip.open(input_path, "rt", encoding="utf-8")
    return input_path.open(encoding="utf-8")


def _find_collection_file(root: Path, collection: str) -> Path:
    direct_candidates = [
        root / f"{collection}.jsonl.gz",
        root / f"{collection}.jsonl",
        root / collection / "train.jsonl.gz",
        root / collection / "train.jsonl",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    recursive_candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".jsonl", ".gz"}
    ]
    collection_candidates = [
        path
        for path in recursive_candidates
        if collection in {part.lower() for part in path.parts}
        or path.name.lower().startswith(collection)
    ]
    train_candidates = [
        path
        for path in collection_candidates
        if "test" not in {part.lower() for part in path.parts}
    ]
    candidates = train_candidates or collection_candidates
    if not candidates:
        raise FileNotFoundError(f"Could not find HH collection {collection!r} under {root}")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _normalize_source_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            return [_find_collection_file(path, collection) for collection in HH_TRAIN_COLLECTIONS]
        return [path]
    return [Path(path) for path in paths]


def _source_split(source_file: str) -> str:
    if source_file.startswith("harmless"):
        return "harmless"
    if source_file.startswith("helpful"):
        return "helpful"
    return "unknown"


def parse_hh_conversation(conversation: str) -> tuple[str, str] | None:
    matches = list(_TURN_RE.finditer(conversation))
    if len(matches) != 2:
        return None
    if matches[0].group(1) != "Human" or matches[1].group(1) != "Assistant":
        return None

    instruction = conversation[matches[0].end() : matches[1].start()].strip()
    response = conversation[matches[1].end() :].strip()
    if not instruction or not response:
        return None
    return instruction, response


def _load_hh_file(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with _open_text(path) as f:
        for line_number, line in enumerate(f, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue

            row = json.loads(line)
            if "chosen" not in row or "rejected" not in row:
                raise ValueError(f"Expected chosen and rejected in {path}:{line_number}")

            chosen = parse_hh_conversation(str(row["chosen"]))
            rejected = parse_hh_conversation(str(row["rejected"]))
            if chosen is None or rejected is None:
                continue

            chosen_instruction, chosen_response = chosen
            rejected_instruction, rejected_response = rejected
            if chosen_instruction != rejected_instruction:
                continue

            records.append(
                {
                    "instruction": chosen_instruction,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "source_file": path.name,
                    "source_path": str(path),
                    "line_number": line_number,
                    "split": _source_split(path.name),
                }
            )
    return records


def load_hh_preference_data(
    paths: str | Path | Iterable[str | Path],
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when set.")

    records: list[dict[str, Any]] = []
    for path in _normalize_source_paths(paths):
        remaining = None if max_records is None else max_records - len(records)
        if remaining is not None and remaining <= 0:
            break
        records.extend(_load_hh_file(path=path, max_records=remaining))

    if not records:
        raise ValueError(f"No single-turn HH preference examples found in {paths}")
    return records
