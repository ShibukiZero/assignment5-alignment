from __future__ import annotations

import re


_MMLU_ANSWER_PATTERNS = [
    re.compile(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*[\(\[]?\s*([ABCD])\s*[\)\]]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:option|choice)\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;]?(?=\s|$)", re.IGNORECASE),
]

_GSM8K_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])"
)


def parse_mmlu_response(mmlu_example: dict, model_output: str) -> str | None:
    """Parse a model MMLU answer into one of A/B/C/D, or None if ambiguous."""
    del mmlu_example

    for pattern in _MMLU_ANSWER_PATTERNS:
        match = pattern.search(model_output)
        if match:
            return match.group(1).upper()
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a GSM8K response by returning the last numeric string it contains."""
    matches = _GSM8K_NUMBER_PATTERN.findall(model_output)
    if not matches:
        return None

    prediction = matches[-1].replace(",", "")
    if "." in prediction:
        whole, fraction = prediction.split(".", maxsplit=1)
        if fraction and set(fraction) == {"0"}:
            return whole
    return prediction
