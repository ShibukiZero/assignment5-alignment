from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


ALPACA_SFT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n"
    "### Response:\n{response}"
)


def _open_text(path: str | Path) -> TextIO:
    input_path = Path(path)
    if input_path.suffix == ".gz":
        return gzip.open(input_path, "rt", encoding="utf-8")
    return input_path.open(encoding="utf-8")


def read_prompt_response_jsonl(
    path: str | Path,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with _open_text(path) as f:
        for line_number, line in enumerate(f, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if "prompt" not in record or "response" not in record:
                raise ValueError(f"Expected prompt and response in {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No prompt/response records found in {path}")
    return records


def format_alpaca_sft_example(prompt: str, response: str) -> str:
    return ALPACA_SFT_TEMPLATE.format(prompt=prompt, response=response)


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
        max_records: int | None = None,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be positive.")

        records = read_prompt_response_jsonl(dataset_path, max_records=max_records)
        if shuffle:
            random.Random(0).shuffle(records)

        token_ids: list[int] = []
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for document packing.")

        for record in records:
            document = format_alpaca_sft_example(
                prompt=str(record["prompt"]),
                response=str(record["response"]),
            )
            token_ids.extend(tokenizer.encode(document, add_special_tokens=True))
            token_ids.append(eos_token_id)

        self.seq_length = seq_length
        self.token_ids = token_ids
        self.num_sequences = max((len(token_ids) - 1) // seq_length, 0)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0 or i >= self.num_sequences:
            raise IndexError(i)
        start = i * self.seq_length
        end = start + self.seq_length
        return {
            "input_ids": torch.tensor(self.token_ids[start:end], dtype=torch.long),
            "labels": torch.tensor(self.token_ids[start + 1 : end + 1], dtype=torch.long),
        }


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
