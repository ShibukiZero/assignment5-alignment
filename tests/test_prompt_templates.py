from cs336_alignment.prompt_templates import (
    ALPACA_SFT_PROMPT_PATH,
    ZERO_SHOT_SYSTEM_PROMPT_PATH,
    format_alpaca_sft_prompt,
    format_generation_prompt,
    format_zero_shot_prompt,
)


def test_format_zero_shot_prompt_uses_official_prompt_file():
    instruction = "Answer with a single word."
    expected = ZERO_SHOT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").format(
        instruction=instruction
    )
    assert format_zero_shot_prompt(instruction) == expected
    assert expected.startswith("# Instruction\n")
    assert expected.endswith("# Answer:\n```\n")


def test_format_alpaca_sft_prompt_uses_official_prompt_file():
    instruction = "Summarize the passage."
    response = "A concise summary."
    expected = ALPACA_SFT_PROMPT_PATH.read_text(encoding="utf-8").format(
        instruction=instruction,
        response=response,
    )
    assert format_alpaca_sft_prompt(instruction, response) == expected


def test_format_generation_prompt_selects_prompt_family():
    instruction = "Classify the example."
    assert format_generation_prompt(instruction, "zero_shot") == format_zero_shot_prompt(
        instruction
    )
    assert format_generation_prompt(instruction, "alpaca_sft") == format_alpaca_sft_prompt(
        instruction
    )
