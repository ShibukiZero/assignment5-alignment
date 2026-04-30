from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_DIR = Path(__file__).with_name("prompts")
ZERO_SHOT_SYSTEM_PROMPT_PATH = PROMPT_DIR / "zero_shot_system_prompt.prompt"
ALPACA_SFT_PROMPT_PATH = PROMPT_DIR / "alpaca_sft.prompt"
PROMPT_FORMAT_CHOICES = ("zero_shot", "alpaca_sft")


@lru_cache(maxsize=None)
def read_prompt_template(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def zero_shot_system_prompt_template() -> str:
    return read_prompt_template(ZERO_SHOT_SYSTEM_PROMPT_PATH)


def alpaca_sft_prompt_template() -> str:
    return read_prompt_template(ALPACA_SFT_PROMPT_PATH)


def format_zero_shot_prompt(instruction: str) -> str:
    return zero_shot_system_prompt_template().format(instruction=instruction)


def format_alpaca_sft_prompt(instruction: str, response: str = "") -> str:
    return alpaca_sft_prompt_template().format(
        instruction=instruction,
        response=response,
    )


def format_generation_prompt(instruction: str, prompt_format: str) -> str:
    if prompt_format == "zero_shot":
        return format_zero_shot_prompt(instruction)
    if prompt_format == "alpaca_sft":
        return format_alpaca_sft_prompt(instruction)
    raise ValueError(f"Unknown prompt format: {prompt_format}")
